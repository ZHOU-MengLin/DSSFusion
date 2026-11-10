import argparse
import logging
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader

from arguments import args
from utils.dataset import Fusion_dataset, CityScapes
from utils.loss import Fusionloss, double_OhemCELoss, int_loss_with_mask, int_loss
from module.FusionNet import FusionNet, SegmentMI
from module.model_TII import BiSeNet
from test import run_fusion
from utils.logger import setup_logger
from utils.optimizer import Optimizer
from utils.utils import *
warnings.filterwarnings('ignore')


def train_seg(train_args, logger): # 分割训练
    logger_step = log_training(args.S_epoch, train_args.n_iter, logger)
    loss_save = AverageMeters(3)
    for epoch in range(0, args.S_epoch):
        loss_save.reset()
        train_bar = tqdm(train_args.seg_loder)
        for it, (im, lb, _) in enumerate(train_bar):
            # print(f"Index: {it}, Names: {_}")
            im, lb = im.cuda(), lb.cuda().squeeze(1)
            train_args.optim.zero_grad()
            out, mid = train_args.net(im)
            loss, lossp, loss2 = train_args.criteria_seg(out, mid, lb, 0.75)
            loss.backward()

            train_args.optim.step()

            loss_save.append([loss, lossp, loss2])
            bar, lr = loss_save.avgs(), train_args.optim.lr
            msg = logger_step(epoch, it, total=bar[0], lossp=bar[1], loss2=bar[2], lr=lr)
            train_bar.set_description(msg)
        if epoch % 10 == 0:
            save_model(train_args.net, f'model/Fusion', f'n{num}_e{epoch}', 'seg')


def train_fusion(train_args, num, logger): # 融合训练
    logger_step = log_training(args.F_epoch, train_args.n_iter, logger, train_args.Res)
    logger.info('Start training the fusion model...')

    mask_dir = f'Dataset/Fusion/mask_f/MSRS_black_{num}'
    os.makedirs(mask_dir, exist_ok=True)

    for epoch in range(train_args.Res, args.F_epoch):
        train_bar = tqdm(train_args.train_loader)

        loss_save = AverageMeters(5)
        for it, (img_vi, img_ir, label, name) in enumerate(train_bar):
            # print(f"Index: {it}, Names: {name}")
            img_vi, img_ir = img_vi.cuda(), img_ir.cuda()
            img_vi_ycrcb = RGB2YCrCb(img_vi)

            if num >= 1:
                mask_f = SegmentMI(img_vi, img_ir, train_args.segmodel, img_vi_ycrcb)
                for k in range(len(name)):
                    for k in range(len(name)):
                        image_np = mask_f[k].squeeze().cpu().numpy()
                        image_np = (image_np * 255).astype(np.uint8)
                        image = Image.fromarray(image_np)
                        save_path = os.path.join(mask_dir, name[k])
                        image.save(save_path)
                img_f_gray = train_args.fusionmodel(img_vi_ycrcb, img_ir, mask_f)
            else:
                img_f_gray = train_args.fusionmodel(img_vi_ycrcb, img_ir)

            img_f = YCrCb2RGB(torch.cat((img_f_gray, img_vi_ycrcb[:, 1:, :, :]), dim=1))
            img_f[img_f > 1] = 1
            img_f[img_f < 0] = 0

            # fusion loss
            label = label.cuda()
            grad_loss = train_args.criteria_fusion(img_vi_ycrcb, img_ir, img_f_gray)
            if num > 0:
                out, mid = train_args.segmodel(img_f)
                seg_loss, _, _ = train_args.criteria_seg(out, mid, label, 0.1)
                mask_loss = int_loss_with_mask(label, img_vi_ycrcb, img_ir, img_f_gray)
                loss_total = 10 * grad_loss + mask_loss + seg_loss
            else:
                mask_loss = int_loss(img_vi_ycrcb, img_ir, img_f_gray)
                loss_total = 10 * grad_loss + mask_loss
                seg_loss = torch.zeros_like(loss_total)  #初始化 seg_loss 为 0

            train_args.optimizer.zero_grad()
            loss_total.backward()
            train_args.optimizer.step()
            train_args.scheduler.step()

            loss_save.append([loss_total, mask_loss, grad_loss, seg_loss])
            bar, lr = loss_save.avgs(), train_args.optimizer.state_dict()['param_groups'][0]['lr']
            msg = logger_step(epoch, it, total=bar[0], mask=bar[1], grad=bar[2], seg=bar[3], lr=lr)
            train_bar.set_description(msg)
        if epoch % 3 == 0:
            save_model(train_args.fusionmodel, f'model/Fusion', f'n{num}_e{epoch}')

def custom_collate(batch):
    images, labels, fnames = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels, fnames  # 返回图像、标签和文件名


def run_train_seg(num, logger=None):  # 配置参数
    parser = argparse.ArgumentParser()
    train_args = parser.parse_args()

    # 加载数据集
    dataset = CityScapes(num=num, cropsize=args.cropsize, mode='train')
    train_args.seg_loder = DataLoader(dataset, batch_size=args.S_batch_size, shuffle=True,
                                      num_workers=args.S_num_workers, pin_memory=True, drop_last=True,collate_fn=custom_collate)
    # 加载分割网络模型
    segmodel = BiSeNet(n_classes=9)
    if num >= 1:
        load_model_path = f'model/Fusion/seg_model_{num - 1}.pth'
        segmodel.load_state_dict(torch.load(load_model_path))
        print(f'Load Segmentation Model < .{load_model_path.split("_data")[-1]} >')
    segmodel = loadnet(segmodel, 'train')
    train_args.net = segmodel

    # 加载损失函数
    n_min = args.S_batch_size * args.cropsize[0] * args.cropsize[1] // 16
    train_args.criteria_seg = double_OhemCELoss(thresh=args.S_score_thres, n_min=n_min, ignore_lb=255)

    # 优化器
    train_args.n_iter = len(train_args.seg_loder)  # 一个epoch中的迭代次数
    train_args.iter_nums = args.S_epoch * train_args.n_iter  # 这一轮训练中要完成的steps
    train_args.it_start = num * train_args.iter_nums  # 这一轮训练的起始step
    train_args.optim = Optimizer(model=train_args.net, lr0=args.S_lr, momentum=0.9, wd=5e-4,
                                 warmup_steps=args.S_warmup_steps, max_iter=train_args.iter_nums * 4,
                                 warmup_start_lr=args.S_warmup_start_lr, power=0.9, it=train_args.it_start)

    # 训练模型
    train_seg(train_args, logger)
    # 保存模型
    save_model(train_args.net, f'model/Fusion', num, 'seg')


def run_train_fusion(num, logger=None):
    parser = argparse.ArgumentParser(description='S2Fusion parameters')
    train_args = parser.parse_args()

    # 加载数据集
    train_dataset = Fusion_dataset('train', args.inf_path, args.vis_path, args.lab_path)
    train_args.train_loader = DataLoader(dataset=train_dataset, batch_size=args.F_batch_size, shuffle=True,
                                         num_workers=args.F_num_workers, pin_memory=True, drop_last=True, )
    train_args.n_iter = len(train_args.train_loader)

    # 加载融合模型--FusionNet
    fusionmodel = FusionNet()
    if num >= 2:
        fusion_model_path = f'model/Fusion/fusion_model_{num - 1}.pth'
        fusionmodel.load_state_dict(torch.load(fusion_model_path))
        print(f'Load fusion model <fusion_model_{num - 1}.pth> to continue training')
    train_args.fusionmodel = loadnet(fusionmodel, 'train')
    train_args.optimizer = torch.optim.Adam([{'params': train_args.fusionmodel.parameters(),
                                              'initial_lr': args.F_lr}], lr=args.F_lr)

    # 加载分割模型--BiSeNet
    segmodel = BiSeNet(n_classes=9)
    if num >= 1:
        seg_model_path = f'model/Fusion/seg_model_{num - 1}.pth'
        segmodel.load_state_dict(torch.load(seg_model_path))
        print(f'Load segmentation model <seg_model_{num - 1}.pth> to continue training')
    train_args.segmodel = loadnet(segmodel, 'test')

    # 学习率优化器
    MaxEpoch2Set = args.F_epoch
    train_args.Res = 0
    # MaxEpoch2Set = args.F_epoch * 3 if num != 0 else args.F_epoch
    # train_args.Res = (num - 1) * args.F_epoch if num != 0 else 0
    train_args.scheduler = get_scheduler(train_args.optimizer, MaxEpoch2Set, train_args.n_iter, resume=train_args.Res)

    # 加载损失函数
    train_args.criteria_seg = double_OhemCELoss() if num > 0 else None
    train_args.criteria_fusion = Fusionloss()
    # 训练模型
    train_fusion(train_args, num, logger)
    # 保存模型
    save_model(train_args.fusionmodel, 'model/Fusion', num, 'fusion')



if __name__ == "__main__":
    setup_seed(15)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger()
    setup_logger('./logs')
    logger.info(f'start training\n')
    start = time.time()

    for num in range(4):

        print('*' * 20)
        print(f'num:{num} start training')

        print(f'num:{num}/4 fusion training start')
        run_train_fusion(num, logger)
        logger.info(f'num:{num}/4 fusion training is done\n')

        print(f'num:{num}/4 fusion image start')
        run_fusion('train', num)
        logger.info(f'num:{num} fusion image is done\n')

        print(f'num:{num} seg training start')
        run_train_seg(num, logger)
        logger.info(f'num:{num} seg training is done\n')
    notification(start)

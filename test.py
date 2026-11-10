# coding:utf-8

import argparse
import os.path
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import Fusion_dataset
from utils.utils import *
from arguments import args
from module.model_TII import BiSeNet
from module.FusionNet import FusionNet, SegmentMI


def run_fusion(type_, num, end=-1):
    fusion_model_path = f'model/Fusion/fusion_model_{num}.pth'
    fused_dir = f'Dataset/Fusion/train/MSRS_black_{num}'
    os.makedirs(fused_dir, exist_ok=True)
    # 加载网络
    fusionmodel = FusionNet()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = loadnet(fusionmodel, 'test')

    segmodel = BiSeNet(n_classes=9)
    if num >= 1:
        load_model_path = f'model/Fusion/seg_model_{num - 1}.pth'
        segmodel.load_state_dict(torch.load(load_model_path))
        print(f'Load Segmentation Model < .{load_model_path.split("_data")[-1]} >')
    segmodel = loadnet(segmodel, 'train')

    # 数据集
    test_dataset = Fusion_dataset(type_, args.inf_path, args.vis_path, args.lab_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.F_batch_size_, shuffle=False,
                             num_workers=args.F_num_workers_, pin_memory=True, drop_last=False, )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_bar):
            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            if num > 0:
                seg = SegmentMI(images_vis, images_ir, segmodel, images_vis_ycrcb)
                logits = fusionmodel(images_vis_ycrcb, images_ir, seg)
            else:
                logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat((logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]), dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image[fusion_image > 1] = 1
            fusion_image[fusion_image < 0] = 0

            fused_image = fusion_image.permute(0, 2, 3, 1).contiguous()
            fused_image = (fused_image - torch.min(fused_image)) / (torch.max(fused_image) - torch.min(fused_image))
            fused_image = np.uint8((255.0 * fused_image).cpu())
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
            if 0 < end < it * args.F_batch_size:
                break
    print(f'Fusion Done! save to {fused_dir}')


def main(seg_model_path, fusion_model_path, ir_dir, vi_dir, save_dir):
    fusionmodel = FusionNet()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = loadnet(fusionmodel, 'test')

    segmodel = BiSeNet(n_classes=9)
    segmodel.load_state_dict(torch.load(seg_model_path))
    segmodel = loadnet(segmodel, 'test')

    # 参数打印
    para = sum([np.prod(list(p.size())) for p in fusionmodel.parameters()])
    Total_params = 0
    for param in fusionmodel.parameters():
        Total_params += np.prod(param.size())
    print(f'Model {fusionmodel._get_name()}  params: {para * 4 / 1000 / 1000:4f}M, Total params: {Total_params}')

    # 数据集加载
    test_dataset = Fusion_dataset('test', ir_path=ir_dir, vi_path=vi_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.F_batch_size_,
        shuffle=False,
        num_workers=args.F_num_workers_,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    import time
    time_space = []

    # 开始测试
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis, img_ir = img_vis.cuda(), img_ir.cuda()
            image_vis_ycrcb = RGB2YCrCb(img_vis)
            st = time.time()
            mask_f = SegmentMI(img_vis, img_ir, segmodel, image_vis_ycrcb)
            fused_img = fusionmodel(image_vis_ycrcb, img_ir, mask_f)
            time_space.append(time.time() - st)

            fusion_ycrcb = torch.cat((fused_img, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]), dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image[fusion_image > 1] = 1
            fusion_image[fusion_image < 0] = 0

            fused_image = fusion_image.permute(0, 2, 3, 1).contiguous()
            fused_image = (fused_image - torch.min(fused_image)) / (torch.max(fused_image) - torch.min(fused_image))
            fused_image = np.uint8((255.0 * fused_image).cpu())
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(save_dir, name[k])
                image.save(save_path)
        print(f'Fusion Done! save to {save_dir}')
    print(f'Fusion Time: {np.mean(time_space)/args.F_batch_size_:4f}±{np.var(time_space)/args.F_batch_size_:4f}s')


def path_porcess(args):
    if not os.path.exists(args.F_model_path) or not os.path.exists(args.S_model_path):
        print('Warning: model path is not exist, please check the path!')
        exit(0)
    if not os.path.exists(args.ir_dir) or not os.path.exists(args.vi_dir):
        print('Warning: image path is not exist, please check the path!')
        exit(0)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--F_model_path', default='model/Fusion/fusion_model_3.pth')
    parser.add_argument('--S_model_path', default='model/Fusion/seg_model_3.pth')
    # dataset (MSRS by default)
    parser.add_argument('--vi_dir', default='./datasets/MSRS/Visible')
    parser.add_argument('--ir_dir', default='./datasets/MSRS/Infrared')
    parser.add_argument('--save_dir', default='./results/fusion')
    args = parser.parse_args()

    path_porcess(args)
    main(seg_model_path=args.S_model_path, fusion_model_path=args.F_model_path,
         ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir)

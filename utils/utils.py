import math
import torch
import numpy as np
import random
import os
import time
from kornia.metrics import AverageMeter
import datetime
from PIL import Image
import torch.nn.functional as F


def save_model(model, path, add, kind='fusion'):
    # 保存模型
    fusion_model_file = os.path.join(path, f'{kind}_model_{add}.pth')
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), fusion_model_file)


def RGB2YCrCb(input_im):
    # print(f"张量形状: {list(input_im.size())}, 元素总数: {input_im.numel()}")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(input_im.device)
    out = (temp.reshape(list(input_im.size())[0],
                        list(input_im.size())[2],
                        list(input_im.size())[3],
                        3, ).transpose(1, 3).transpose(2, 3))
    return out


def YCrCb2RGB(input_im):
    # print(f"张量形状: {list(input_im.size())}, 元素总数: {input_im.numel()}")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).to(input_im.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(input_im.device)
    temp = (im_flat + bias).mm(mat)
    out = (temp.reshape(list(input_im.size())[0],
                        list(input_im.size())[2],
                        list(input_im.size())[3],
                        3, ).transpose(1, 3).transpose(2, 3))
    return out


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_log(args):
    print_list = ['resume', 'resume_file_name', 'int_loss_type', 'loss_weight',
                  'no_fea_epochs', 'feature_loss_weight', 'forward_grad', 'mlp_cat']
    if os.path.exists(args.save_file_path) is False:
        os.makedirs(args.save_file_path)
    new_file_path = os.path.join(args.save_file_path, 'log.txt')
    time_str = time.strftime("%m-%d %H:%M", time.localtime())
    with open(new_file_path, 'a') as f:
        f.write(f'===== time:{time_str} =====\n')
        for arg in vars(args):
            s = format(arg, '<25') + ':' + format(str(getattr(args, arg)), '<')
            print(s) if arg in print_list else None
            f.write(s + '\n')
        f.write('\n\n')


class AverageMeters(AverageMeter):
    def __init__(self, num_counters=1):
        super().__init__()
        self.num = num_counters
        self.counters = [AverageMeter() for _ in range(self.num)]
        self.save = [[] for _ in range(self.num)]

    def append(self, x, n=0):
        if n != 0:
            self.save[n].append(x.item())
        else:
            length = min(self.num, len(x))
            for i in range(length):
                self.save[i].append(x[i].item())
                self.counters[i].update(x[i].item())

    def avgs(self, num=0):
        num = self.num if num == 0 else num
        if num == 0:
            num = self.num
        else:
            num = min(num, self.num)
        x = []
        for i in range(num):
            x.append(self.counters[i].avg)
        return x

    @property
    def log(self):
        return self.save


def get_scheduler(optimizer, epoch, length, lr_m='index', resume=0, warmup_steps=200, time=2, h=0.25):
    def lambda_rule(step):
        nonlocal a, b
        if step < warmup_steps:
            return step / warmup_steps  # 在warmup期间线性增加学习率
        else:
            if lr_m == 'cosine':  # 使用余弦退火来调整学习率
                amplitude = a * step + b
                cos_decay = 0.5 * (1 + math.cos(math.pi * time * (step - warmup_steps) / (max_steps - warmup_steps)))
                return cos_decay * amplitude
            elif lr_m == 'index':  # 使用指数衰减来调整学习率
                index_decay = 0.9996 ** (step - warmup_steps)
                return index_decay

    max_steps = epoch * length
    cur_step = resume * length - 1
    if lr_m == 'cosine':
        time = time * 2 - 1
        a = (1 - h) / (warmup_steps - max_steps)
        b = (h * warmup_steps - max_steps) / (warmup_steps - max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_step)
    return scheduler


def notification(start_time, string=None):
    use_time = time.time() - start_time
    hours, remainder = divmod(use_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    message = f"程序运行时间：{int(hours)} h {int(minutes)} m {int(seconds)} s"
    print(message)


def log_training(epoch, lenth_iter, logger, res=0):
    st = glob_st = time.time()
    total_epoch = epoch
    n_iter = lenth_iter
    logger_ = logger

    def log_step(epo, it, **kwargs):
        nonlocal st, glob_st, total_epoch, n_iter, logger_, res
        ed = time.time()

        glob_t_intv = ed - glob_st  # 运行的时间
        now_it = n_iter * epo + it + 1  # 现在是第几个it
        make_it = now_it - res * n_iter  # 完成的it  （当重启时现在的it与完成的it不一致）
        single_it_t = (glob_t_intv / make_it)  # 单个it的时间
        et_it = n_iter * total_epoch - now_it  # 剩余it的个数
        eta = str(datetime.timedelta(seconds=int(et_it * single_it_t)))  # 剩余的时间

        # 取kwargs中的键和值，拼接成字符串
        result = f'e: {epo}/{total_epoch}, it: {now_it + res * n_iter}/{n_iter * total_epoch}, '
        for key, value in kwargs.items():
            result += f"{key}: {value:.4f}, " if key != 'lr' else f"{key}: {value:.6f}, "
        msg = result + f'eta: {eta}'
        st = ed

        # 10it -> log
        if it % 10 == 0 or it == 0 or it == lenth_iter - 1:
            logger_.info(msg)
        return msg

    return log_step


def loadnet(net, m):
    net = net.cuda()
    if m == 'train':
        net.train()
    elif m == 'test':
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
    else:
        raise ValueError('net mode error')
    return net


# tensor to PIL Image
def tensor2img(img, is_norm=True):
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    return img.astype(np.uint8)


def save_img_single(img, name):
    img = tensor2img(img, is_norm=True)
    img = Image.fromarray(img)
    img.save(name)


def interp(x, t):
    return F.interpolate(x, size=(t.size(2), t.size(3)), mode='bilinear', align_corners=False)

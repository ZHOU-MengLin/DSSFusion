#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# from utils.utils import save_img_single


def double_OhemCELoss(thresh=0.7, n_min=8 * 640 * 480 // 8, ignore_lb=255):
    criteria_p = OhemCELoss(thresh, n_min, ignore_lb)
    criteria_16 = OhemCELoss(thresh, n_min, ignore_lb)

    def forward(out, mid, label, alpha):
        label = label.long()
        lossp = criteria_p(out, label)
        loss2 = criteria_16(mid, label)
        seg_loss = lossp + alpha * loss2
        return seg_loss, lossp, loss2

    return forward


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


# class NormalLoss(nn.Module):
#     def __init__(self, ignore_lb=255, *args, **kwargs):
#         super(NormalLoss, self).__init__()
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
#
#     def forward(self, logits, labels):
#         N, C, H, W = logits.size()
#         loss = self.criteria(logits, labels)
#         return torch.mean(loss)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        loss_grad = F.l1_loss(torch.max(y_grad, ir_grad), generate_img_grad)
        return loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def int_loss(vir, ir, fuse):
    vi = vir[:, :1]
    loss_int = F.l1_loss(torch.max(ir, vi), fuse)
    return loss_int


def int_loss_with_mask(seg, vir, ir, fuse):
    alpha = 0.6 # 默认0.6
    vi = vir[:, :1]
    # seg mask
    seg = seg.unsqueeze(1)
    seg_bg = torch.where(seg == 0, 1.0, 0.0)
    seg_fg = 1 - seg_bg

    abs_img = torch.abs(fuse - torch.max(vi, ir))
    # foreground loss
    loss_f = (abs_img * seg_fg).sum() / (seg_fg.sum() + 0.00001)
    # background loss
    loss_b = (abs_img * seg_bg).sum() / (seg_bg.sum() + 0.00001)
    # total loss
    loss = (1 + alpha) * loss_b + (1 - alpha) * loss_f
    return loss


# if __name__ == '__main__':
#     # 生成一个随机的0或1图像
#     seg = torch.randint(0, 2, (1, 1, 100, 100)).cuda()
#     x = torch.randn(1, 1, 100, 100).cuda()
#     x1 = torch.randn(1, 3, 100, 100).cuda()
#     x2 = torch.randn(1, 1, 100, 100).cuda()
#     int_loss_with_mask(seg, x1, x2, x)
#     pass

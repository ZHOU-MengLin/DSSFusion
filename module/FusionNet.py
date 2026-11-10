

# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.TransformerNet import SDIB
# from utils.utils import interp
from module.CBAM_old import DAFB
from utils.utils import YCrCb2RGB
from module.FreqFusion import FreqFusion


# class ConvBnLeakyRelu2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(ConvBnLeakyRelu2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
#                               dilation=dilation, groups=groups)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.init_weight()
#
#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if ly.bias is not None:
#                     nn.init.constant_(ly.bias, 0)
#
#     def forward(self, x):
#         return F.leaky_relu(self.conv(x), negative_slope=0.2)


# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             # nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)


class ConvTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


# class HugeDenseBlock(nn.Module):
#     def __init__(self, channels):
#         super(HugeDenseBlock, self).__init__()
#         self.conv1 = ConvLeakyRelu2d(channels, channels)
#         self.conv2 = ConvLeakyRelu2d(2 * channels, 2 * channels)
#         self.conv3 = ConvLeakyRelu2d(4 * channels, 4 * channels)
#         self.conv4 = ConvLeakyRelu2d(8 * channels, 8 * channels)
#         self.sobelconv = Sobelxy(channels)
#         self.convup = Conv1(channels, channels * 8)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(torch.cat((x, x1), dim=1))
#         x3 = self.conv3(torch.cat((x, x1, x2), dim=1))
#         x4 = self.conv4(torch.cat((x, x1, x2, x3), dim=1))
#         grad = self.convup(self.sobelconv(x))
#         return F.leaky_relu(grad + x4, negative_slope=0.1)


class DenseBlockWithSobel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlockWithSobel, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)

        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        y = x1 + x2
        return F.leaky_relu(y, negative_slope=0.1)


class EncoderNet(nn.Module):
    def __init__(self, channels=(16, 32, 48)):
        super(EncoderNet, self).__init__()
        self.vi_begin = ConvLeakyRelu2d(1, channels[0], kernel_size=5, padding=2)
        self.ir_begin = ConvLeakyRelu2d(1, channels[0], kernel_size=5, padding=2)


        self.vi_net = DenseBlockWithSobel(channels[0], channels[1])
        self.ir_net = DenseBlockWithSobel(channels[0], channels[1])

    def forward(self, vi, ir):
        vis = self.vi_begin(vi)
        inf = self.ir_begin(ir)
        vis = self.vi_net(vis)
        inf = self.ir_net(inf)
        return vis, inf


class Merge(nn.Module):
    def __init__(self, inchannel):
        super().__init__()

        # DAFB
        self.ir_DAFB = DAFB(inchannel)
        self.vi_DAFB = DAFB(inchannel)

        # SDIB
        self.b = SDIB(inchannel, dim=12, depth=4, depths=4, num_heads=8)
        self.b1_conv1 = Conv1(12, inchannel)
        self.b2_conv1 = Conv1(12, inchannel)

    def forward(self, vis_fea, inf_fea, mask_f):
        # # 前景 4dm
        vi_f = self.vi_DAFB(vis_fea)
        ir_f = self.ir_DAFB(inf_fea)

        # 背景
        vi_b, ir_b = self.b(vis_fea, inf_fea)
        vi_b, ir_b = self.b1_conv1(vi_b), self.b2_conv1(ir_b)
        if vi_b.shape[2] != vis_fea.shape[2] or vi_b.shape[3] != vis_fea.shape[3]:
            vi_b = F.interpolate(vi_b, size=vis_fea.shape[2:], mode='bilinear', align_corners=True)
            ir_b = F.interpolate(ir_b, size=inf_fea.shape[2:], mode='bilinear', align_corners=True)
        vi_b, ir_b = vi_b + vis_fea, ir_b + inf_fea

        # 重造
        # mask_f = 0.5
        vis_fea = vi_f * mask_f + vi_b * (1 - mask_f)
        inf_fea = ir_f * mask_f + ir_b * (1 - mask_f)
        # inf_fea = ir_f * mask_f

        # SDIB 有背景没前进
        # vi_b, ir_b = self.b(vis_fea, inf_fea)
        # vi_b, ir_b = self.b1_conv1(vi_b), self.b2_conv1(ir_b)
        # if vi_b.shape[2] != vis_fea.shape[2] or vi_b.shape[3] != vis_fea.shape[3]:
        #     vi_b = F.interpolate(vi_b, size=vis_fea.shape[2:], mode='bilinear', align_corners=True)
        #     ir_b = F.interpolate(ir_b, size=inf_fea.shape[2:], mode='bilinear', align_corners=True)
        # vi_b, ir_b = vi_b + vis_fea, ir_b + inf_fea
        # vis_fea = vis_fea * mask_f + vi_b * (1 - mask_f)
        # inf_fea = inf_fea * mask_f + ir_b * (1 - mask_f)

        # # DAFB 有前景没背景
        # vi_f = self.vi_DAFB(vis_fea)
        # ir_f = self.ir_DAFB(inf_fea)
        # vis_fea = vi_f * mask_f + inf_fea * (1 - mask_f)
        # inf_fea = ir_f * mask_f + inf_fea * (1 - mask_f)

        return vis_fea, inf_fea


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        channels = (16, 32, 48)
        self.encoder = EncoderNet(channels)
        self.freq_fusion = FreqFusion(hr_channels=32, lr_channels=32)
        self.conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.decoder = nn.Sequential(
            ConvLeakyRelu2d(32 * 2, 48),
            ConvLeakyRelu2d(48, 24),
            ConvLeakyRelu2d(24, 12),
            ConvTanh2d(12, 1)
        )
        self.merge = Merge(32)

    def forward(self, image_vi, image_ir, mask_F=None, return_debug=False, mask_mode=None):
        vis_origin = image_vi[:, :1]
        inf_origin = image_ir

        # encode
        vis_fea1, inf_fea1 = self.encoder(vis_origin, inf_origin)

        # vis_fea2 = self.conv2d(vis_fea1)
        # inf_fea2 = self.conv2d(inf_fea1)
        # _, vis_fea1, vis_fea2 = self.freq_fusion(hr_feat=vis_fea1, lr_feat=vis_fea2)
        # _, inf_fea1, inf_fea2 = self.freq_fusion(hr_feat=inf_fea1, lr_feat=inf_fea2)

        # merge
        if mask_F is not None and mask_mode is None:
            vis_fea1, inf_fea1 = self.merge(vis_fea1, inf_fea1, mask_F)
        elif mask_mode is not None:
            vis_fea1, inf_fea1 = self._apply_mask_mode(vis_fea1, inf_fea1, mask_mode, mask_F)

        # decode
        fuse = self.decoder(torch.cat([vis_fea1, inf_fea1], dim=1))

        return fuse

    def _apply_mask_mode(self, vis_fea, inf_fea, mask_mode, mask_F=None):
        B, C, H, W = vis_fea.shape
        device = vis_fea.device

        if mask_mode == 'no_mask':
            return vis_fea, inf_fea

        if mask_mode == 'all_fg':
            mask = torch.ones(B, 1, H, W, device=device)
        elif mask_mode == 'all_bg':
            mask = torch.zeros(B, 1, H, W, device=device)
        elif mask_mode == 'uniform':
            mask = torch.full((B, 1, H, W), 0.5, device=device)
        elif mask_mode == 'random':
            mask = torch.randint(0, 2, (B, 1, H, W), device=device).float()
        elif mask_mode == 'gt_mask':
            mask = mask_F
        elif mask_mode == 'pred_mask':
            mask = mask_F
        else:
            return vis_fea, inf_fea

        return self.merge(vis_fea, inf_fea, mask)


def SegmentMI(vi, ir, net, ycrcb):
    ir = torch.cat((ir, ycrcb[:, 1:, :, :]), dim=1)
    Sir = net(YCrCb2RGB(ir))[0]  # [B, 9, 640, 480]
    Svi = net(vi)[0]

    seg_ir = torch.max(Sir, dim=1, keepdim=True)[1]
    seg_vi = torch.max(Svi, dim=1, keepdim=True)[1]
    mask_f = torch.max(seg_ir, seg_vi)
    mask_f = torch.where(mask_f > 0, 1.0, 0.0)
    return mask_f


if __name__ == '__main__':
    net = FusionNet().cuda()
    seg = torch.randn((2, 1, 640, 480)).cuda()
    x1 = torch.randn((2, 3, 640, 480)).cuda()
    x2 = torch.randn((2, 1, 640, 480)).cuda()
    y = net(x1, x2, seg)
    print(y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class DAFB(nn.Module):
    def __init__(self, channel, reduction=12, spatial_kernel=7):
        super(DAFB, self).__init__()
        self.RB = nn.Sequential(
            ConvLeakyRelu2d(channel, channel, kernel_size=3, padding=1),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )

        # Spatial Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # Channel Attention
        self.SAB = nn.Sequential(
            ConvLeakyRelu2d(channel, channel, kernel_size=spatial_kernel, padding=spatial_kernel // 2),
            nn.Conv2d(channel, channel, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_rb = self.RB(x)
        x_Spatial_Attention = self.mlp(self.avg_pool(x_rb))
        x_Channel_Channel = self.SAB(x_rb)
        f = self.sigmoid(x_Spatial_Attention * x_Channel_Channel)
        x_out = self.relu(f * x_rb + x)
        return x_out


if __name__ == '__main__':
    vi = torch.randn(2, 64, 480, 640)
    ir = torch.randn(2, 64, 480, 640)
    print(vi.size(), ir.size())
    net = DAFB(64)
    x1, x2 = net.forward(ir)
    print(x1.shape,x2.shape)

# class CAM(nn.Module):
#     def __init__(self, channel, reduction, to_end=True, out_channel=None):
#         super(CAM, self).__init__()
#         self.out_channel = channel if out_channel is None else out_channel
#         self.to_end = to_end
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # shared MLP
#         self.mlp = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             nn.LeakyReLU(inplace=False),
#             nn.Conv2d(channel // reduction, self.out_channel, 1, bias=False),
#             nn.BatchNorm2d(self.out_channel)
#         )
#         self.mlp1 = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             nn.LeakyReLU(inplace=False),
#             nn.Conv2d(channel // reduction, self.out_channel, 1, bias=False),
#             nn.BatchNorm2d(self.out_channel)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp1(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         if self.to_end:
#             x = channel_out * x
#             return channel_out, x
#         return channel_out


# class SAM(nn.Module):
#     def __init__(self, kernel_size=7, to_end=True):
#         super(SAM, self).__init__()
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.to_end = to_end
#
#     def forward(self, x):
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         if self.to_end:
#             x = spatial_out * x
#             return spatial_out, x
#         return spatial_out


# class CBAM(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = CAM(channel, reduction)
#         self.spatial_attention = SAM(spatial_kernel)
#
#     def forward(self, x):
#         channel_out, x = self.channel_attention(x)
#         spatial_out, x = self.spatial_attention(x)
#         return spatial_out, x


# class CBAMLayer(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()
#
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x


# class DRB(nn.Module):
#     def __init__(self, channel):
#         super().__init__()
#         self.RB_DRB = nn.ModuleList([
#             nn.Conv2d(channel, channel, kernel_size=9, padding=9 // 2),
#             nn.Conv2d(channel, channel, kernel_size=5, padding=5 // 2),
#             nn.Conv2d(channel, channel, kernel_size=5, padding=5 // 2 + 2, dilation=2),
#             nn.Conv2d(channel, channel, kernel_size=3, padding=3 // 2 + 2, dilation=3),
#             nn.Conv2d(channel, channel, kernel_size=3, padding=3 // 2 + 3, dilation=4),
#         ])
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         for layer in self.RB_DRB:
#             x = x + layer(x) if x is not None else layer(x)
#         return self.relu(x)






if __name__ == '__main__':
    vi = torch.randn(2, 64, 480, 640)
    ir = torch.randn(2, 64, 480, 640)
    net = DAFB(64)
    x1, x2 = net.forward(ir)
    print(x1.shape)

    fused_visible = net(vi)
    fused_infrared = net(ir)

    fused_features = torch.cat([fused_visible, fused_infrared], dim=1)
    print(f"融合特征尺寸: {fused_features.shape}")
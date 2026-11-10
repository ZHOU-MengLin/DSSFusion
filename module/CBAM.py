import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientChannelAttention(nn.Module):
    """高效通道注意力机制 (ECA)"""

    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        # 自适应确定卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 特征描述符 [batch, channels, 1, 1]
        y = self.avg_pool(x)

        # 1D卷积处理通道关系
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 生成通道注意力权重
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上计算最大和平均特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # 拼接特征并应用卷积
        combined = torch.cat([max_out, avg_out], dim=1)
        attention = self.sigmoid(self.conv(combined))

        return x * attention


class DAFB(nn.Module):
    """改进的双注意力融合模块"""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(DAFB, self).__init__()

        # 特征提取分支
        self.feature_extract = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 通道注意力分支
        self.channel_att = EfficientChannelAttention(channels)

        # 空间注意力分支
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 残差连接缩放参数
        self.residual_scale = nn.Parameter(torch.zeros(1))

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 特征提取
        identity = x
        features = self.feature_extract(x)

        # 并行双注意力处理
        channel_attended = self.channel_att(features)
        spatial_attended = self.spatial_att(features)

        # 融合双注意力特征
        fused = torch.cat([channel_attended, spatial_attended], dim=1)
        fused = self.fusion_conv(fused)

        # 残差连接
        out = fused + self.residual_scale * identity
        return self.relu(out)


if __name__ == '__main__':
    vi = torch.randn(2, 64, 480, 640)
    ir = torch.randn(2, 64, 480, 640)

    dafb = DAFB(channels=64)
    x1, _ = dafb.forward(ir)
    print(x1.shape)

    fused_visible = dafb(vi)
    fused_infrared = dafb(ir)

    fused_features = torch.cat([fused_visible, fused_infrared], dim=1)
    print(f"融合特征尺寸: {fused_features.shape}")
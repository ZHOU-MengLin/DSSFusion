import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from math import log2


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


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table0 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        self.relative_position_bias_table1 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv0 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop0 = nn.Dropout(attn_drop)
        self.proj0 = nn.Linear(dim, dim)
        self.proj_drop0 = nn.Dropout(proj_drop)

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table0, std=.02)
        nn.init.trunc_normal_(self.relative_position_bias_table1, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x0, x1, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x0: input features with shape of (num_windows*B, Mh*Mw, C)
            x1: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x0.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv0 = self.qkv0(x0).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv1 = self.qkv1(x1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q0, k0, v0 = qkv0.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q1, k1, v1 = qkv1.unbind(0)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q0, q1 = q0 * self.scale, q1 * self.scale
        # 交换域外信息querry
        attn0 = (q1 @ k0.transpose(-2, -1))
        attn1 = (q0 @ k1.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias0 = self.relative_position_bias_table0[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias1 = self.relative_position_bias_table1[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        relative_position_bias0 = relative_position_bias0.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        relative_position_bias1 = relative_position_bias1.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn0 = attn0 + relative_position_bias0.unsqueeze(0)
        attn1 = attn1 + relative_position_bias1.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn0 = attn0.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn0 = attn0.view(-1, self.num_heads, N, N)
            attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N)

        attn0 = self.softmax(attn0)
        attn0 = self.attn_drop0(attn0)
        attn1 = self.softmax(attn1)
        attn1 = self.attn_drop1(attn1)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x0 = (attn0 @ v0).transpose(1, 2).reshape(B_, N, C)
        x0 = self.proj0(x0)
        x0 = self.proj_drop0(x0)

        x1 = (attn1 @ v1).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj1(x1)
        x1 = self.proj_drop1(x1)
        return x0, x1


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * 2 * mlp_ratio)
        self.mlp0 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        out_features=dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        out_features=dim, act_layer=act_layer, drop=drop)
        self.sobel = Sobelxy(dim)

    def forward(self, x0, x1, attn_mask):
        H, W, C = x0.shape[1], x0.shape[2], x0.shape[3]

        shortcut0 = x0
        shortcut1 = x1
        x0 = self.norm1(x0)  # LayerNorm
        x1 = self.norm2(x1)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x0 = F.pad(x0, (0, 0, pad_l, pad_r, pad_t, pad_b))
            x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x0.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x0 = torch.roll(x0, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x0 = x0
            shifted_x1 = x1
            attn_mask = None

        # partition windows
        x_windows0 = window_partition(shifted_x0, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows1 = window_partition(shifted_x1, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows0 = x_windows0.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows0, attn_windows1 = self.attn(x_windows0, x_windows1, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows0 = attn_windows0.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        attn_windows1 = attn_windows1.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x0 = window_reverse(attn_windows0, self.window_size, Hp, Wp)  # [B, H', W', C]
        shifted_x1 = window_reverse(attn_windows1, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x0 = torch.roll(shifted_x0, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x0 = shifted_x0
            x1 = shifted_x1

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x0 = x0[:, :H, :W, :].contiguous()
            x1 = x1[:, :H, :W, :].contiguous()

        # FFN
        # grad_a = self.sobel(x0.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # grad_b = self.sobel(x1.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        z0 = shortcut0 + self.drop_path0(x0)
        z1 = shortcut1 + self.drop_path1(x1)

        n0 = self.norm3(z0)
        n1 = self.norm2(z1)
        x0 = z0 + self.drop_path0(self.mlp0(n0))
        x1 = z1 + self.drop_path1(self.mlp1(n1))
        return x0, x1


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def create_mask(self, x):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        H, W = x.shape[1], x.shape[2]
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x0, x1):
        attn_mask = self.create_mask(x0)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            x0, x1 = blk(x0, x1, attn_mask)
        return x0, x1


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. x -> [B, H, W, C] -> [B, H/2, W/2, 2*C]
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        # padding
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            B, H, W, C = x.shape
        # [B, H/2, W/2, C]
        x0 = x[:, 0::2, 0::2, :]  # 取偶数行、偶数列的块
        x1 = x[:, 1::2, 0::2, :]  # 取奇数行、偶数列的块
        x2 = x[:, 0::2, 1::2, :]  # 取偶数行、奇数列的块
        x3 = x[:, 1::2, 1::2, :]  # 取奇数行、奇数列的块
        # [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)  # 将这四个块沿着通道方向拼接
        # [B, H/2*W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # 展平成形状为[B, H/2*W/2, 4*C]的张量
        x = self.norm(x)
        # [B, H/2*W/2, 2*C]
        x = self.reduction(x)
        # [B, H/2, W/2, 2*C]
        x = x.view(B, int(H / 2), int(W / 2), 2 * C)
        return x


class PatchReverse(nn.Module):
    r""" PatchMerging 的逆过程，将 [B, H/2, W/2, 2*C]的张量转换成 [B, H, W, C]的张量
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        # x [B, H/2, W/2, 2*C] -> [B, H, W, C]
        B, h, w, c = x.shape
        H = h * 2
        W = w * 2
        C = c // 2
        if C != self.dim:
            raise ValueError(f'x.shape[-1] must be {self.dim * 2}, but get {x.shape[-1]}')
        # [B, H/2*W/2, 2*C]
        x = x.view(B, H // 2 * W // 2, 2 * C)
        x = self.norm(x)
        # [B, H/2*W/2, 4*C]
        x = self.reduction(x)
        # [B, H, W, C]
        x = x.view(B, H, W, C)
        return x


class PatchMergings(nn.Module):
    def __init__(self, dim, depth=8):
        super().__init__()
        self.depth = int(log2(depth))
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(PatchMerging(dim * 2 ** i))

    def forward(self, x):
        # x [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        for i in range(self.depth):
            x = self.layers[i](x)
        return x


class BilinearUp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv = ConvLeakyRelu2d(dim, dim // 2, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.up(x)
        x = self.Conv(x)
        return x


class PatchReverseBlock(nn.Module):
    def __init__(self, dim, depth=8, dtype='BilinearUp'):
        super().__init__()
        self.depth = int(log2(depth))
        self.layers = nn.ModuleList()
        self.dtype = dtype
        if dtype == 'PatchReverse':
            for i in range(self.depth):
                self.layers.append(PatchReverse(dim * 2 ** (self.depth - i - 1)))
        elif dtype == 'BilinearUp':
            for i in range(self.depth):
                self.layers.append(BilinearUp(dim * 2 ** (self.depth - i)))
            # self.layers.append(BilinearUp(dim * 2 ** (self.depth - i)))

    def forward(self, x):
        if self.dtype == 'BilinearUp':
            # x [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        for i in range(self.depth):
            x = self.layers[i](x)
        if self.dtype == 'PatchReverse':
            # x [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class CDST(nn.Module):
    def __init__(self, embed_dim=48, depths=2, num_heads=8):
        super().__init__()
        self.layers = BasicLayer(dim=embed_dim,
                                 depth=depths,
                                 num_heads=num_heads,
                                 window_size=7,
                                 mlp_ratio=4,
                                 qkv_bias=True,
                                 drop=0,
                                 attn_drop=0.1,
                                 drop_path=0.1,
                                 norm_layer=nn.LayerNorm)

    def forward(self, features_vi, features_ir):
        x0, x1 = self.layers(features_vi, features_ir)
        x = torch.cat((x0, x1), dim=-1)
        return x.split(features_vi.shape[-1], dim=-1)


class SDIB(nn.Module):
    def __init__(self, in_channels, dim=12, depth=8, depths=4, num_heads=8):
        super().__init__()
        self.convDown_vi_3x3 = nn.Sequential(
            ConvLeakyRelu2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=1),
            ConvLeakyRelu2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1),
            ConvLeakyRelu2d(in_channels // 2, dim, kernel_size=3, padding=1, stride=1),
        )
        self.convDown_ir_3x3 = nn.Sequential(
            ConvLeakyRelu2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=1),
            ConvLeakyRelu2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1),
            ConvLeakyRelu2d(in_channels // 2, dim, kernel_size=3, padding=1, stride=1),
        )
        self.vi_PMs = PatchMergings(dim=dim, depth=depth).cuda()
        self.ir_PMs = PatchMergings(dim=dim, depth=depth).cuda()
        self.cdst = CDST(embed_dim=dim * depth, depths=depths, num_heads=num_heads).cuda()
        self.vi_PRB = PatchReverseBlock(dim=dim, depth=depth).cuda()
        self.ir_PRB = PatchReverseBlock(dim=dim, depth=depth).cuda()

    def forward(self, vi_feas, ir_feas):
        x1 = self.convDown_vi_3x3(vi_feas)
        x2 = self.convDown_ir_3x3(ir_feas)
        vi_feas = self.vi_PMs(x1)
        ir_feas = self.ir_PMs(x2)
        vi_feas, ir_feas = self.cdst(vi_feas, ir_feas)
        y1 = self.vi_PRB(vi_feas)
        y2 = self.ir_PRB(ir_feas)
        return y1, y2

if __name__ == '__main__':
    dim = 8
    depth = 4
    x1 = torch.randn(1, 48, 480, 640).cuda()
    x2 = torch.randn(1, 48, 480, 640).cuda()
    model = SDIB(in_channels=48, dim=dim, depth=depth).cuda()
    y1, y2 = model(x1, x2)
    print(y1.shape)

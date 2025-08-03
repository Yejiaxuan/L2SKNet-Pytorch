import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0, bias=False),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0, bias=False),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)


class LLSKM(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LLSKM, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$      
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        
        # The output feature of our Diff LSFM block
        # $\textbf{Y} = {{\mathcal{W}}_s (\textbf{X})} = \mathcal{W}_{sum}(\textbf{X}) - {\mathcal{W}}(\textbf{X}) + \theta_c (\textbf{X})\circ {\mathcal{W}_{c}}{(\textbf{X})}$
        return out_center - out_normal + theta * out_offset

class LLSKM_d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False):
        super(LLSKM_d, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # The output feature of our Diff LSFM block
        # $\textbf{Y} = {{\mathcal{W}}_s (\textbf{X})} = \mathcal{W}_{sum}(\textbf{X}) - {\mathcal{W}}(\textbf{X}) + \theta_c (\textbf{X})\circ {\mathcal{W}_{c}}{(\textbf{X})}$
        return out_center - out_normal + theta * out_offset

class LLSKM_1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LLSKM_1D, self).__init__()
        self.conv_1xn = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=(stride, stride),
                                  padding=(0, padding), bias=bias)
        self.conv_nx1 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=(stride, stride),
                                  padding=(padding, 0), bias=bias)
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

    def forward(self, x):
        theta = self.attn(x)

        # 1D分解卷积：先1xn，再nx1
        out_1xn_normal = self.conv_1xn(x)
        out_nx1_normal = self.conv_nx1(out_1xn_normal)
        
        # 获取卷积核权重
        kernel_1xn = self.conv_1xn.weight  # [out_channels, in_channels, 1, kernel_size]
        kernel_nx1 = self.conv_nx1.weight  # [out_channels, in_channels, kernel_size, 1]
        
        # 重构nxn卷积核：通过外积操作
        # kernel_1xn: [C, C, 1, K] -> [C, C, K]
        # kernel_nx1: [C, C, K, 1] -> [C, C, K]
        kernel_1xn_reshaped = kernel_1xn.squeeze(2)  # [C, C, K]
        kernel_nx1_reshaped = kernel_nx1.squeeze(3)  # [C, C, K]
        
        # 通过外积重构nxn核
        nxn_kernel = torch.zeros((kernel_1xn.shape[0], kernel_1xn.shape[1], self.kernel_size, self.kernel_size), 
                                device=x.device, dtype=x.dtype)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                nxn_kernel[:, :, i, j] = kernel_nx1_reshaped[:, :, i] * kernel_1xn_reshaped[:, :, j]

        # 计算sum kernel
        nxn_kernel_sum = nxn_kernel.sum(2).sum(2)  # [C, C]
        nxn_kernel_sum = nxn_kernel_sum[:, :, None, None]  # [C, C, 1, 1]
        out_center = F.conv2d(x, weight=nxn_kernel_sum, bias=self.conv_1xn.bias, stride=self.stride,
                              padding=0, groups=self.groups)

        # 计算center kernel
        center_idx = self.kernel_size // 2
        nxn_center_kernel = nxn_kernel[:, :, center_idx, center_idx]  # [C, C]
        nxn_center_kernel = nxn_center_kernel[:, :, None, None]  # [C, C, 1, 1]
        out_offset = F.conv2d(x, weight=nxn_center_kernel, bias=self.conv_1xn.bias, stride=self.stride,
                              padding=0, groups=self.groups)

        # LLSKM输出
        out = out_center - out_nx1_normal + theta * out_offset

        return out
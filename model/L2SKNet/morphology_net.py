"""
MorphologyNet: 可微分形态学知识迁移模块
将传统红外小目标检测的形态学操作（Top-Hat、Max-Median、LoG等）用深度学习重构
实现「区域增强 → 噪声抑制 → 判别阈值」三阶段pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableThreshold(nn.Module):
    """可学习软阈值模块，使用sigmoid平滑过渡"""
    def __init__(self, channels, init_value=0.5):
        super(LearnableThreshold, self).__init__()
        self.threshold = nn.Parameter(torch.ones(1, channels, 1, 1) * init_value)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 动态阈值：sigmoid(threshold) * max_value
        max_val = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)[0].unsqueeze(-1)
        dynamic_thresh = self.sigmoid(self.threshold) * max_val
        
        # 软阈值：使用sigmoid平滑过渡
        gate = torch.sigmoid(10 * (x - dynamic_thresh))  # 10是陡峭度参数
        return x * gate


class DifferentiableTopHat(nn.Module):
    """可微分Top-Hat变换：用于区域增强"""
    def __init__(self, channels, kernel_size=7):
        super(DifferentiableTopHat, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 形态学开运算的可微分近似：先腐蚀后膨胀
        # 腐蚀：depthwise conv + min pooling近似
        self.erosion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, 
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 膨胀：depthwise conv + max pooling近似  
        self.dilation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2,
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 初始化为均值滤波器
        self._init_morphology_weights()
        
    def _init_morphology_weights(self):
        """初始化形态学核为均值滤波器"""
        with torch.no_grad():
            weight_value = 1.0 / (self.kernel_size * self.kernel_size)
            self.erosion[0].weight.fill_(weight_value)
            self.dilation[0].weight.fill_(weight_value)
    
    def forward(self, x):
        # Top-Hat = 原图 - 开运算(原图)
        # 开运算 = 膨胀(腐蚀(x))
        eroded = -F.relu(-self.erosion(x))  # 近似min操作
        opened = F.relu(self.dilation(eroded))  # 近似max操作
        top_hat = x - opened
        return F.relu(top_hat)  # 只保留正值


class DifferentiableMaxMedian(nn.Module):
    """可微分Max-Median滤波：用于噪声抑制"""
    def __init__(self, channels, window_size=5):
        super(DifferentiableMaxMedian, self).__init__()
        self.window_size = window_size
        
        # Max滤波：使用可学习的权重近似
        self.max_filter = nn.Conv2d(channels, channels, window_size, 
                                   padding=window_size//2, groups=channels, bias=False)
        
        # Median滤波的可微分近似：使用多个不同权重的卷积组合
        self.median_filters = nn.ModuleList([
            nn.Conv2d(channels, channels, window_size, padding=window_size//2, 
                     groups=channels, bias=False) for _ in range(3)
        ])
        
        self.fusion = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
        self._init_filters()
        
    def _init_filters(self):
        """初始化滤波器权重"""
        with torch.no_grad():
            # Max filter: 中心权重大
            center = self.window_size // 2
            for i in range(self.max_filter.weight.size(0)):
                self.max_filter.weight[i, 0].fill_(0.1)
                self.max_filter.weight[i, 0, center, center] = 0.7
            
            # Median filters: 不同的权重分布
            for idx, filter_layer in enumerate(self.median_filters):
                weight = torch.randn_like(filter_layer.weight) * 0.1
                weight[:, :, center, center] += 0.5
                filter_layer.weight.copy_(weight)
    
    def forward(self, x):
        max_out = self.max_filter(x)
        median_outs = [f(x) for f in self.median_filters]
        
        # 融合所有输出
        combined = torch.cat([max_out] + median_outs, dim=1)
        out = self.fusion(combined)
        return self.bn(out)


class DifferentiableLoG(nn.Module):
    """可微分LoG（拉普拉斯高斯）算子：用于边缘检测和小目标增强"""
    def __init__(self, channels, sigma=1.0):
        super(DifferentiableLoG, self).__init__()
        self.channels = channels
        self.sigma = sigma
        
        # 高斯滤波
        self.gaussian = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        
        # 拉普拉斯算子
        self.laplacian = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        
        self._init_log_kernel()
        
    def _init_log_kernel(self):
        """初始化LoG核"""
        with torch.no_grad():
            # 高斯核
            gaussian_kernel = self._get_gaussian_kernel(5, self.sigma)
            for i in range(self.channels):
                self.gaussian.weight[i, 0] = gaussian_kernel
            
            # 拉普拉斯核
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1], 
                [0, -1, 0]
            ], dtype=torch.float32)
            for i in range(self.channels):
                self.laplacian.weight[i, 0] = laplacian_kernel
    
    def _get_gaussian_kernel(self, size, sigma):
        """生成高斯核"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    def forward(self, x):
        # 先高斯平滑，再拉普拉斯
        smoothed = self.gaussian(x)
        log_out = self.laplacian(smoothed)
        return torch.abs(log_out)  # 取绝对值突出边缘


class MorphologyNet(nn.Module):
    """
    统一的多尺度形态学网络
    通过调整scales参数控制复杂度：
    - 高分辨率层：scales=[3, 5, 7] (多尺度)
    - 中分辨率层：scales=[3, 5] (中等尺度)  
    - 低分辨率层：scales=[5] (单尺度)
    """
    def __init__(self, channels, scales=[3, 5, 7]):
        super(MorphologyNet, self).__init__()
        self.scales = scales
        
        # 阶段1：多尺度区域增强
        self.multi_tophat = nn.ModuleList([
            DifferentiableTopHat(channels, kernel_size=scale) for scale in scales
        ])
        
        # 多尺度LoG算子（用于边缘增强）
        self.multi_log = nn.ModuleList([
            DifferentiableLoG(channels, sigma=scale/3.0) for scale in scales
        ])
        
        # 阶段1特征融合
        total_features = len(scales) * 2 + 1  # tophat + log + 原始特征
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(channels * total_features, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 阶段2：噪声抑制
        self.max_median = DifferentiableMaxMedian(channels, window_size=5)
        
        # 阶段2特征融合
        self.stage2_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),  # enhanced + denoised
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 阶段3：判别阈值
        self.threshold = LearnableThreshold(channels, init_value=0.3)
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # 阶段1：多尺度区域增强
        features = [x]  # 包含原始特征
        
        # 多尺度Top-Hat
        for tophat in self.multi_tophat:
            features.append(tophat(x))
            
        # 多尺度LoG
        for log_filter in self.multi_log:
            features.append(log_filter(x))
        
        # 融合增强特征
        combined = torch.cat(features, dim=1)
        enhanced = self.stage1_fusion(combined)
        
        # 阶段2：噪声抑制
        denoised = self.max_median(enhanced)
        stage2_features = [enhanced, denoised]
        combined_stage2 = torch.cat(stage2_features, dim=1)
        refined = self.stage2_fusion(combined_stage2)
        
        # 阶段3：判别阈值
        thresholded = self.threshold(refined)
        
        # 残差连接
        output = x + self.alpha * thresholded
        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 16, 64, 64).to(device)
    
    # 测试不同尺度配置的多尺度形态学网络
    print("=== 统一多尺度形态学网络测试 ===")
    
    # 高分辨率层配置：多尺度
    morph_high = MorphologyNet(channels=16, scales=[3, 5, 7]).to(device)
    out_high = morph_high(x)
    print(f"高分辨率层 (scales=[3,5,7]) output shape: {out_high.shape}")
    
    # 中分辨率层配置：中等尺度
    morph_mid = MorphologyNet(channels=16, scales=[3, 5]).to(device)
    out_mid = morph_mid(x)
    print(f"中分辨率层 (scales=[3,5]) output shape: {out_mid.shape}")
    
    # 低分辨率层配置：单尺度
    morph_low = MorphologyNet(channels=16, scales=[5]).to(device)
    out_low = morph_low(x)
    print(f"低分辨率层 (scales=[5]) output shape: {out_low.shape}")
    
    # 参数量对比
    high_params = sum(p.numel() for p in morph_high.parameters())
    mid_params = sum(p.numel() for p in morph_mid.parameters())
    low_params = sum(p.numel() for p in morph_low.parameters())
    
    print(f"\n=== 参数量对比 ===")
    print(f"高分辨率层 (3尺度): {high_params:,} 参数")
    print(f"中分辨率层 (2尺度): {mid_params:,} 参数")
    print(f"低分辨率层 (1尺度): {low_params:,} 参数")
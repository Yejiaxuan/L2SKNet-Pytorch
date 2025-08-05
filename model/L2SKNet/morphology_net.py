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
    """可学习阈值模块，替代传统固定阈值"""
    def __init__(self, channels, init_value=0.5):
        super(LearnableThreshold, self).__init__()
        self.threshold = nn.Parameter(torch.ones(1, channels, 1, 1) * init_value)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 动态阈值：sigmoid(threshold) * max_value
        max_val = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)[0].unsqueeze(-1)
        dynamic_thresh = self.sigmoid(self.threshold) * max_val
        return torch.where(x > dynamic_thresh, x, torch.zeros_like(x))


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
    形态学知识迁移网络
    实现传统「区域增强 → 噪声抑制 → 判别阈值」三阶段pipeline
    """
    def __init__(self, channels, enable_tophat=True, enable_maxmedian=True, enable_log=True):
        super(MorphologyNet, self).__init__()
        self.enable_tophat = enable_tophat
        self.enable_maxmedian = enable_maxmedian  
        self.enable_log = enable_log
        
        # 阶段1：区域增强
        if enable_tophat:
            self.top_hat = DifferentiableTopHat(channels, kernel_size=7)
        if enable_log:
            self.log_filter = DifferentiableLoG(channels, sigma=1.0)
            
        # 阶段2：噪声抑制
        if enable_maxmedian:
            self.max_median = DifferentiableMaxMedian(channels, window_size=5)
        
        # 阶段3：判别阈值
        self.threshold = LearnableThreshold(channels, init_value=0.3)
        
        # 特征融合 - 阶段1
        stage1_channels = channels
        if enable_tophat:
            stage1_channels += channels
        if enable_log:
            stage1_channels += channels
            
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(stage1_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合 - 阶段2（如果启用噪声抑制）
        if enable_maxmedian:
            self.stage2_fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),  # enhanced + denoised
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        
        # 残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        features = [x]  # 原始特征
        
        # 阶段1：区域增强
        if self.enable_tophat:
            tophat_out = self.top_hat(x)
            features.append(tophat_out)
            
        if self.enable_log:
            log_out = self.log_filter(x)
            features.append(log_out)
        
        # 融合增强特征
        enhanced = torch.cat(features, dim=1)
        enhanced = self.stage1_fusion(enhanced)
        
        # 阶段2：噪声抑制
        if self.enable_maxmedian:
            denoised = self.max_median(enhanced)
            features_stage2 = [enhanced, denoised]
            combined = torch.cat(features_stage2, dim=1)
            combined = self.stage2_fusion(combined)
        else:
            combined = enhanced
            
        # 阶段3：判别阈值
        thresholded = self.threshold(combined)
        
        # 残差连接：原始特征 + α * 形态学特征
        output = x + self.alpha * thresholded
        
        return output


class MorphologyNetLite(nn.Module):
    """
    轻量级形态学网络，用于计算资源受限的情况
    """
    def __init__(self, channels):
        super(MorphologyNetLite, self).__init__()
        
        # 简化的Top-Hat
        self.simple_tophat = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 简化的阈值
        self.threshold = LearnableThreshold(channels, init_value=0.2)
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.05))
        
    def forward(self, x):
        # 简单的增强操作
        enhanced = self.simple_tophat(x)
        thresholded = self.threshold(enhanced)
        
        # 残差连接
        output = x + self.alpha * thresholded
        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试完整版
    morph_net = MorphologyNet(channels=16).to(device)
    x = torch.randn(2, 16, 64, 64).to(device)
    out = morph_net(x)
    print(f"MorphologyNet output shape: {out.shape}")
    
    # 测试轻量版
    morph_lite = MorphologyNetLite(channels=16).to(device)
    out_lite = morph_lite(x)
    print(f"MorphologyNetLite output shape: {out_lite.shape}")
    
    # 参数量对比
    total_params = sum(p.numel() for p in morph_net.parameters())
    lite_params = sum(p.numel() for p in morph_lite.parameters())
    print(f"MorphologyNet params: {total_params}")
    print(f"MorphologyNetLite params: {lite_params}")
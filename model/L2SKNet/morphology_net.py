import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableThreshold(nn.Module):
    """可学习的软阈值模块，用于自适应特征筛选"""
    def __init__(self, channels, init_value=0.5):
        super(LearnableThreshold, self).__init__()
        self.threshold = nn.Parameter(torch.ones(1, channels, 1, 1) * init_value)
        self.temperature_raw = nn.Parameter(torch.zeros(1))  # 控制阈值锐度
        
    def forward(self, x):
        # 使用LogSumExp计算软最大值，避免梯度消失
        beta = 10.0
        x_flat = x.contiguous().view(x.size(0), x.size(1), -1)
        lse = torch.logsumexp(beta * x_flat, dim=2, keepdim=True)
        max_val = (lse / beta).unsqueeze(-1)
        
        # 动态阈值：基于特征最大值的比例
        dynamic_thresh = torch.sigmoid(self.threshold) * max_val
        tau = 1.0 + 19.0 * torch.sigmoid(self.temperature_raw)
        gate = torch.sigmoid(tau * (x - dynamic_thresh))
        return x * gate


class DifferentiableTopHat(nn.Module):
    """可微分Top-Hat变换，用于小目标区域增强"""
    def __init__(self, channels, kernel_size=7):
        super(DifferentiableTopHat, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def _min_pool_2d_with_reflect(self, x, kernel_size, stride=1):
        """腐蚀操作：邻域最小值"""
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        return -F.max_pool2d(-x_padded, kernel_size, stride, padding=0)
    
    def _max_pool_2d_with_reflect(self, x, kernel_size, stride=1):
        """膨胀操作：邻域最大值"""
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        return F.max_pool2d(x_padded, kernel_size, stride, padding=0)
    
    def forward(self, x):
        # 形态学开运算：腐蚀后膨胀，去除小噪声
        eroded = self._min_pool_2d_with_reflect(x, self.kernel_size, stride=1)
        opened = self._max_pool_2d_with_reflect(eroded, self.kernel_size, stride=1)
        
        # Top-Hat变换：原图减去开运算结果，突出小目标
        top_hat = x - opened
        return F.relu(self.scale * top_hat + self.bias)


class DifferentiableMaxMedian(nn.Module):
    """Max-Median滤波器，结合最大值和中值滤波进行噪声抑制"""
    def __init__(self, channels, window_size=5):
        super(DifferentiableMaxMedian, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        
        # 可解释混合门：让网络学会在Max与Median之间选择
        self.mix = nn.Parameter(torch.zeros(1, channels, 1, 1))  # 初值≈0.5
        
        # 可学习的β参数，控制中值滤波的锐度
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        
    def _true_max_filter(self, x):
        """最大值滤波：保持目标强度"""
        padding = self.window_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        return F.max_pool2d(x_padded, self.window_size, stride=1, padding=0)
    
    def _approximate_median_filter(self, x):
        """软中值滤波：抑制脉冲噪声"""
        padding = self.window_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        B, C, H, W = x.shape
        patches = F.unfold(x_padded, self.window_size, padding=0)
        patches = patches.contiguous().view(B, C, self.window_size*self.window_size, H*W)
        
        # 可学习的β参数，映射到[5, 50]范围
        beta = 5.0 + 45.0 * torch.sigmoid(self.beta_raw)
        center = patches.mean(dim=2, keepdim=True)
        weights = F.softmax(-beta * torch.abs(patches - center), dim=2)
        median_values = (weights * patches).sum(dim=2)
        
        return median_values.contiguous().view(B, C, H, W)
    
    def forward(self, x):
        max_out = self._true_max_filter(x)
        median_out = self._approximate_median_filter(x)
        
        # 可解释混合门：在Max与Median之间进行选择
        lambda_c = torch.sigmoid(self.mix)  # [0,1]
        out = lambda_c * median_out + (1 - lambda_c) * max_out
        return out


class DifferentiableLoG(nn.Module):
    """拉普拉斯高斯算子，用于边缘检测和小目标增强"""
    def __init__(self, channels, sigma=1.0):
        super(DifferentiableLoG, self).__init__()
        sigma = max(0.6, min(float(sigma), 1.2))  # 限制sigma范围
        self.channels = channels
        self.sigma = sigma
        
        # 高斯平滑核和拉普拉斯算子
        self.gaussian = nn.Conv2d(channels, channels, 5, padding=0, groups=channels, bias=False)
        self.laplacian = nn.Conv2d(channels, channels, 3, padding=0, groups=channels, bias=False)
        
        # 可学习的正负权重，保留符号信息
        self.pos_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.neg_weight = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        
        self._init_log_kernel()
        
    def _init_log_kernel(self):
        """初始化LoG核参数"""
        with torch.no_grad():
            # 初始化高斯核
            gaussian_kernel = self._get_gaussian_kernel(5, self.sigma)
            for i in range(self.channels):
                self.gaussian.weight[i, 0] = gaussian_kernel
            
            # 初始化拉普拉斯核
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1], 
                [0, -1, 0]
            ], dtype=torch.float32)
            for i in range(self.channels):
                self.laplacian.weight[i, 0] = laplacian_kernel
        
        # 固定核参数，保持可解释性
        self.gaussian.weight.requires_grad_(False)
        self.laplacian.weight.requires_grad_(False)
    
    def _get_gaussian_kernel(self, size, sigma):
        """生成2D高斯核"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    def forward(self, x):
        # 先高斯平滑
        x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
        smoothed = self.gaussian(x_padded)
        
        # 再拉普拉斯边缘检测
        smoothed_padded = F.pad(smoothed, (1, 1, 1, 1), mode='reflect')
        log_out = self.laplacian(smoothed_padded)
        
        # 分别处理正负响应，保留更多信息
        pos_part = F.relu(log_out) * self.pos_weight
        neg_part = F.relu(-log_out) * self.neg_weight
        
        return pos_part + neg_part


class MorphologyNet(nn.Module):
    """多尺度形态学网络：区域增强 → 噪声抑制 → 判别阈值"""
    def __init__(self, channels, scales=[3, 5, 7]):
        super(MorphologyNet, self).__init__()
        self.scales = scales
        
        # 阶段1：多尺度Top-Hat变换，增强不同尺寸的小目标
        self.multi_tophat = nn.ModuleList([
            DifferentiableTopHat(channels, kernel_size=scale) for scale in scales
        ])
        
        # 阶段1：多尺度LoG算子，检测不同尺度的边缘特征
        self.multi_log = nn.ModuleList([
            DifferentiableLoG(channels, sigma=scale/3.0) for scale in scales
        ])
        
        # 阶段1特征融合：原始特征 + Top-Hat + LoG
        total_features = len(scales) * 2 + 1
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(channels * total_features, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 阶段2：Max-Median滤波器，抑制噪声
        self.max_median = DifferentiableMaxMedian(channels, window_size=5)
        
        # 阶段2特征融合：增强特征 + 去噪特征
        self.stage2_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 阶段3：自适应阈值，最终特征筛选
        self.threshold = LearnableThreshold(channels, init_value=0.3)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 残差连接权重
        
    def forward(self, x):
        # 阶段1：多尺度区域增强
        features = [x]  # 保留原始特征
        
        # 多尺度Top-Hat变换
        for tophat in self.multi_tophat:
            features.append(tophat(x))
            
        # 多尺度LoG边缘检测
        for log_filter in self.multi_log:
            features.append(log_filter(x))
        
        # 融合所有增强特征
        combined = torch.cat(features, dim=1)
        enhanced = self.stage1_fusion(combined)
        
        # 阶段2：噪声抑制
        denoised = self.max_median(enhanced)
        stage2_features = [enhanced, denoised]
        combined_stage2 = torch.cat(stage2_features, dim=1)
        refined = self.stage2_fusion(combined_stage2)
        
        # 阶段3：自适应判别阈值
        thresholded = self.threshold(refined)
        
        # 残差连接，保持特征稳定性
        alpha_smooth = torch.sigmoid(self.alpha)
        output = x + alpha_smooth * thresholded
        return output


if __name__ == "__main__":
    # 测试不同尺度配置的形态学网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 16, 64, 64).to(device)
    
    # 高分辨率层：多尺度配置
    morph_high = MorphologyNet(channels=16, scales=[3, 5, 7]).to(device)
    out_high = morph_high(x)
    print(f"高分辨率层 output shape: {out_high.shape}")
    
    # 中分辨率层：中等尺度配置
    morph_mid = MorphologyNet(channels=16, scales=[3, 5]).to(device)
    out_mid = morph_mid(x)
    print(f"中分辨率层 output shape: {out_mid.shape}")
    
    # 低分辨率层：单尺度配置
    morph_low = MorphologyNet(channels=16, scales=[5]).to(device)
    out_low = morph_low(x)
    print(f"低分辨率层 output shape: {out_low.shape}")
    
    # 参数量统计
    high_params = sum(p.numel() for p in morph_high.parameters())
    mid_params = sum(p.numel() for p in morph_mid.parameters())
    low_params = sum(p.numel() for p in morph_low.parameters())
    
    print(f"参数量: 高分辨率 {high_params:,}, 中分辨率 {mid_params:,}, 低分辨率 {low_params:,}")
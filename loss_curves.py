#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NUDT-SIRST训练日志Loss曲线绘制工具
自动解析log文件夹中的NUDT-SIRST训练日志，绘制所有loss曲线
"""

import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log_file(log_path):
    """解析单个日志文件，提取epoch和loss数据"""
    epochs = []
    losses = []
    
    # 从文件名提取模型名称
    filename = Path(log_path).stem
    model_name = '_'.join(filename.split('_')[:-6])  # 移除日期时间部分
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 使用正则表达式匹配epoch和loss信息
    pattern = r'Epoch---(\d+),\s*total_loss---([0-9.]+)'
    matches = re.findall(pattern, content)
    
    for epoch_str, loss_str in matches:
        epochs.append(int(epoch_str))
        losses.append(float(loss_str))
    
    return epochs, losses, model_name

def main():
    """主函数：绘制所有NUDT-SIRST模型的loss曲线"""
    log_dir = Path('log')
    
    # 创建输出文件夹
    output_dir = Path('loss_curves')
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有NUDT-SIRST日志文件
    nudt_logs = [f for f in log_dir.glob('*.txt') if 'NUDT-SIRST' in f.name]
    
    if not nudt_logs:
        print("No NUDT-SIRST log files found!")
        return
    
    # 设置字体和图形参数
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 颜色列表
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 创建综合对比图
    plt.figure(figsize=(12, 8))
    
    print(f"Found {len(nudt_logs)} NUDT-SIRST log files:")
    print(f"Output directory: {output_dir.absolute()}")
    
    for i, log_file in enumerate(nudt_logs):
        epochs, losses, model_name = parse_log_file(log_file)
        
        if epochs and losses:
            color = colors[i % len(colors)]
            label = model_name.replace('NUDT-SIRST_', '')
            plt.plot(epochs, losses, color=color, linewidth=2, 
                    label=label, marker='o', markersize=3)
            
            # 打印统计信息
            min_loss = min(losses)
            min_epoch = epochs[losses.index(min_loss)]
            final_loss = losses[-1]
            print(f"  {label}: Final={final_loss:.6f}, Min={min_loss:.6f} (Epoch {min_epoch})")
    
    plt.title('NUDT-SIRST Training Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存综合对比图
    comparison_path = output_dir / 'nudt_sirst_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {comparison_path}")
    plt.show()
    
    # 为每个模型创建单独的详细图
    for log_file in nudt_logs:
        epochs, losses, model_name = parse_log_file(log_file)
        
        if epochs and losses:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
            
            # 计算统计信息
            min_loss = min(losses)
            min_epoch = epochs[losses.index(min_loss)]
            final_loss = losses[-1]
            initial_loss = losses[0]
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            
            # 添加统计信息文本框
            stats_text = f'Initial: {initial_loss:.6f}\n'
            stats_text += f'Min: {min_loss:.6f} (Epoch {min_epoch})\n'
            stats_text += f'Final: {final_loss:.6f}\n'
            stats_text += f'Reduction: {loss_reduction:.2f}%'
            
            plt.text(0.02, 0.98, stats_text, 
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            clean_name = model_name.replace('NUDT-SIRST_', '')
            plt.title(f'Training Loss Curve - {clean_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Total Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存单个模型图
            output_name = output_dir / f"{clean_name}_loss_curve.png"
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to: {output_name}")
            plt.show()
    
if __name__ == "__main__":
    main()
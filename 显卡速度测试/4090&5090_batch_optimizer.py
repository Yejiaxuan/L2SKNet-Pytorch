#!/usr/bin/env python3
"""
RTX 4090专用batch size优化器
针对高端GPU的大batch size测试
"""

import time
import torch
import os
import json
from torch.utils.data import DataLoader
from net import Net
from utils.utils import seed_pytorch
import numpy as np

from utils.datasets import NUDTSIRSTSetLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_batch_config(dataset_name, dataset_class, data_dir, batch_size, num_workers, test_batches=8):
    """测试单个batch_size和num_workers配置"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        dataset = dataset_class(base_dir=data_dir, mode='trainval')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        net = Net(model_name='L2SKNet_UNet').to(device)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
        
        # 预热
        data_iter = iter(dataloader)
        for _ in range(2):
            try:
                img, gt_mask = next(data_iter)
                img, gt_mask = img.to(device), gt_mask.to(device)
                pred = net.forward(img)
                loss = net.loss(pred, gt_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except StopIteration:
                data_iter = iter(dataloader)
                continue
        
        # 正式测试
        batch_times = []
        data_times = []
        compute_times = []
        
        for i in range(test_batches):
            try:
                data_start = time.time()
                img, gt_mask = next(data_iter)
                img, gt_mask = img.to(device), gt_mask.to(device)
                data_time = time.time() - data_start
                data_times.append(data_time)
                
                compute_start = time.time()
                pred = net.forward(img)
                loss = net.loss(pred, gt_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                compute_time = time.time() - compute_start
                compute_times.append(compute_time)
                
                total_time = data_time + compute_time
                batch_times.append(total_time)
                
            except StopIteration:
                data_iter = iter(dataloader)
                continue
        
        avg_batch_time = np.mean(batch_times)
        avg_data_time = np.mean(data_times)
        avg_compute_time = np.mean(compute_times)
        throughput = batch_size / avg_batch_time
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'throughput': throughput,
            'avg_batch_time': avg_batch_time,
            'avg_data_time': avg_data_time,
            'avg_compute_time': avg_compute_time,
            'gpu_memory': gpu_memory,
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'success': False, 
            'error': str(e), 
            'throughput': 0,
            'batch_size': batch_size,
            'num_workers': num_workers
        }

def find_optimal_batch_sizes_rtx4090():
    """针对RTX 4090的batch size优化"""
    print("RTX 4090 Batch Size优化测试")
    print("=" * 80)
    
    datasets = {
        'NUDT-SIRST': {
            'class': NUDTSIRSTSetLoader,
            'dir': './data/NUDT-SIRST/',
            'image_size': 256
        }
    }
    
    # RTX 4090专用测试配置 - 可以测试更大的batch size
    batch_sizes = [
        # 小batch size
        2, 3, 4, 5, 6, 7, 8, 
        # 中等batch size  
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        # 大batch size (4090专属)
        36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128
    ]
    num_workers_list = [0, 1, 2, 3, 4, 8]  # 4090可以测试更多workers
    
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU内存: {gpu_memory:.1f} GB")
        
        if gpu_memory > 20:  # 确认是高端GPU
            print("检测到高端GPU，启用大batch size测试")
        else:
            print("检测到中低端GPU，限制batch size范围")
            batch_sizes = [b for b in batch_sizes if b <= 64]
    
    print(f"测试batch_size: {len(batch_sizes)}个值")
    print(f"测试num_workers: {num_workers_list}")
    print()
    
    all_results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*20} 测试 {dataset_name} {'='*20}")
        print(f"图像大小: {config['image_size']}x{config['image_size']}")
        print("-" * 60)
        
        dataset_results = []
        
        for num_workers in num_workers_list:
            print(f"\nnum_workers = {num_workers}:")
            
            for i, batch_size in enumerate(batch_sizes):
                print(f"  [{i+1:2d}/{len(batch_sizes)}] BS={batch_size:3d}: ", end="", flush=True)
                
                result = test_batch_config(
                    dataset_name, 
                    config['class'], 
                    config['dir'], 
                    batch_size, 
                    num_workers
                )
                
                if result['success']:
                    print(f"{result['throughput']:6.1f} samples/s, "
                          f"时间={result['avg_batch_time']:.3f}s, "
                          f"GPU={result['gpu_memory']:.1f}GB")
                    dataset_results.append(result)
                else:
                    print(f"失败: {result['error']}")
                    # 如果连续失败，可能是batch size太大，跳过后续更大的值
                    if "out of memory" in result['error'].lower():
                        print(f"    GPU内存不足，跳过更大的batch size")
                        break
        
        # 分析结果
        if dataset_results:
            dataset_results.sort(key=lambda x: x['throughput'], reverse=True)
            all_results[dataset_name] = dataset_results
            
            print(f"\n{dataset_name} 最佳配置:")
            print("-" * 40)
            
            # 显示前10个最佳配置
            top_configs = dataset_results[:10]
            for i, result in enumerate(top_configs, 1):
                print(f"{i:2d}. BS={result['batch_size']:3d}, Workers={result['num_workers']} -> "
                      f"{result['throughput']:6.1f} samples/s, "
                      f"时间={result['avg_batch_time']:.3f}s, "
                      f"GPU={result['gpu_memory']:.1f}GB")
            
            # 分析不同num_workers的最佳配置
            print(f"\n各num_workers最佳配置:")
            for workers in num_workers_list:
                worker_results = [r for r in dataset_results if r['num_workers'] == workers]
                if worker_results:
                    best = worker_results[0]
                    print(f"  Workers={workers}: BS={best['batch_size']:3d} -> {best['throughput']:6.1f} samples/s")
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("RTX 4090 最终推荐配置")
    print("=" * 80)
    
    final_recommendations = {}
    
    for dataset_name, results in all_results.items():
        if results:
            best_overall = results[0]
            
            print(f"\n{dataset_name}:")
            print(f"  最佳配置: BS={best_overall['batch_size']}, Workers={best_overall['num_workers']}")
            print(f"  最佳性能: {best_overall['throughput']:.1f} samples/s")
            print(f"  GPU内存使用: {best_overall['gpu_memory']:.1f} GB")
            
            # 计算相比RTX 5060的提升
            rtx5060_performance = {
                'NUDT-SIRST': 62.5
            }
            
            if dataset_name in rtx5060_performance:
                speedup = best_overall['throughput'] / rtx5060_performance[dataset_name]
                print(f"  相比RTX 5060提升: {speedup:.1f}倍")
            
            final_recommendations[dataset_name] = {
                'batch_size': best_overall['batch_size'],
                'num_workers': best_overall['num_workers'],
                'throughput': round(best_overall['throughput'], 1),
                'gpu_memory': round(best_overall['gpu_memory'], 1)
            }
    
    # 保存结果
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    with open(f'rtx4090_optimization_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    with open('rtx4090_optimal_config.json', 'w', encoding='utf-8') as f:
        json.dump(final_recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: rtx4090_optimization_results_{timestamp}.json")
    print("推荐配置已保存到: rtx4090_optimal_config.json")
    
    return final_recommendations

if __name__ == '__main__':
    seed_pytorch(42)
    find_optimal_batch_sizes_rtx4090()
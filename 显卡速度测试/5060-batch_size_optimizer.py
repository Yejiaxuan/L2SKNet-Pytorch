#!/usr/bin/env python3
"""
Batch Size优化器 - 专门测试不同batch_size对训练速度的影响
只测试num_workers=0和1，重点测试更多batch_size值
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
from utils.datasets import IRSTD1KSetLoader
from utils.datasets import SIRSTAugSetLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_batch_config(dataset_name, dataset_class, data_dir, batch_size, num_workers, test_batches=8):
    """测试单个batch_size和num_workers配置"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 创建数据集和加载器
        dataset = dataset_class(base_dir=data_dir, mode='trainval')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        # 创建模型
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
                # 数据加载时间
                data_start = time.time()
                img, gt_mask = next(data_iter)
                img, gt_mask = img.to(device), gt_mask.to(device)
                data_time = time.time() - data_start
                data_times.append(data_time)
                
                # 计算时间
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
        
        # 计算统计信息
        avg_batch_time = np.mean(batch_times)
        avg_data_time = np.mean(data_times)
        avg_compute_time = np.mean(compute_times)
        throughput = batch_size / avg_batch_time
        
        # GPU内存使用
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

def find_optimal_batch_sizes():
    """寻找每个数据集的最优batch_size"""
    print("寻找最优Batch Size配置")
    print("=" * 80)
    
    # 数据集配置
    datasets = {
        'NUDT-SIRST': {
            'class': NUDTSIRSTSetLoader,
            'dir': './data/NUDT-SIRST/',
            'image_size': 256
        },
        'IRSTD-1K': {
            'class': IRSTD1KSetLoader,
            'dir': './data/IRSTD-1K/',
            'image_size': 512
        },
        'SIRST-aug': {
            'class': SIRSTAugSetLoader,
            'dir': './data/sirst_aug/',
            'image_size': 256
        }
    }
    
    # 测试的batch_size值 - 更全面的测试
    batch_sizes = [8]
    num_workers_list = [1]  # 测试0, 1, 2
    
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
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
                print(f"  [{i+1:2d}/{len(batch_sizes)}] BS={batch_size:2d}: ", end="", flush=True)
                
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
        
        # 分析结果
        if dataset_results:
            # 按吞吐量排序
            dataset_results.sort(key=lambda x: x['throughput'], reverse=True)
            all_results[dataset_name] = dataset_results
            
            print(f"\n{dataset_name} 最佳配置:")
            print("-" * 40)
            
            # 显示前10个最佳配置
            top_configs = dataset_results[:10]
            for i, result in enumerate(top_configs, 1):
                print(f"{i:2d}. BS={result['batch_size']:2d}, Workers={result['num_workers']} -> "
                      f"{result['throughput']:6.1f} samples/s, "
                      f"时间={result['avg_batch_time']:.3f}s, "
                      f"数据={result['avg_data_time']:.3f}s, "
                      f"计算={result['avg_compute_time']:.3f}s, "
                      f"GPU={result['gpu_memory']:.1f}GB")
            
            # 分别显示num_workers=0, 1, 2的最佳配置
            for workers in [0, 1, 2]:
                worker_results = [r for r in dataset_results if r['num_workers'] == workers]
                if worker_results:
                    best = worker_results[0]
                    print(f"\nnum_workers={workers}最佳: BS={best['batch_size']}, {best['throughput']:.1f} samples/s")
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("最终推荐配置")
    print("=" * 80)
    
    final_recommendations = {}
    
    for dataset_name, results in all_results.items():
        if results:
            best_overall = results[0]
            
            # 分别找num_workers=0, 1, 2的最佳配置
            best_worker0 = next((r for r in results if r['num_workers'] == 0), None)
            best_worker1 = next((r for r in results if r['num_workers'] == 1), None)
            best_worker2 = next((r for r in results if r['num_workers'] == 2), None)
            
            print(f"\n{dataset_name}:")
            print(f"  总体最佳: BS={best_overall['batch_size']}, Workers={best_overall['num_workers']} -> {best_overall['throughput']:.1f} samples/s")
            
            if best_worker0:
                print(f"  Workers=0最佳: BS={best_worker0['batch_size']} -> {best_worker0['throughput']:.1f} samples/s")
            if best_worker1:
                print(f"  Workers=1最佳: BS={best_worker1['batch_size']} -> {best_worker1['throughput']:.1f} samples/s")
            if best_worker2:
                print(f"  Workers=2最佳: BS={best_worker2['batch_size']} -> {best_worker2['throughput']:.1f} samples/s")
            
            final_recommendations[dataset_name] = {
                'overall_best': {
                    'batch_size': best_overall['batch_size'],
                    'num_workers': best_overall['num_workers'],
                    'throughput': round(best_overall['throughput'], 1)
                },
                'worker0_best': {
                    'batch_size': best_worker0['batch_size'] if best_worker0 else None,
                    'throughput': round(best_worker0['throughput'], 1) if best_worker0 else None
                } if best_worker0 else None,
                'worker1_best': {
                    'batch_size': best_worker1['batch_size'] if best_worker1 else None,
                    'throughput': round(best_worker1['throughput'], 1) if best_worker1 else None
                } if best_worker1 else None,
                'worker2_best': {
                    'batch_size': best_worker2['batch_size'] if best_worker2 else None,
                    'throughput': round(best_worker2['throughput'], 1) if best_worker2 else None
                } if best_worker2 else None
            }
    
    # 保存详细结果
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # 保存完整结果
    with open(f'batch_size_optimization_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存推荐配置
    with open('optimal_batch_config.json', 'w', encoding='utf-8') as f:
        json.dump(final_recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: batch_size_optimization_results_{timestamp}.json")
    print("推荐配置已保存到: optimal_batch_config.json")
    
    return final_recommendations

if __name__ == '__main__':
    seed_pytorch(42)
    find_optimal_batch_sizes()
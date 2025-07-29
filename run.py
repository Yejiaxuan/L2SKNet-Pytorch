#!/usr/bin/env python3

import subprocess
import time
import threading
import re
from datetime import datetime
from tqdm import tqdm

def run_training_with_progress(cmd, model, dataset):
    """运行训练并显示epoch进度条"""
    total_epochs = 400  # 默认epoch数量
    
    # 创建进度条
    pbar = tqdm(total=total_epochs, desc=f'{model} on {dataset}', unit='epoch')
    
    # 使用Popen实时获取输出并更新进度条
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             universal_newlines=True, bufsize=1)
    
    # 实时读取输出并更新进度条
    for line in process.stdout:
        # 匹配epoch信息: "Epoch---123, total_loss---0.123456"
        epoch_match = re.search(r'Epoch---(\d+), total_loss---([\d.]+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            loss = float(epoch_match.group(2))
            
            # 更新进度条
            pbar.n = current_epoch
            pbar.set_postfix({'loss': f'{loss:.6f}'})
            pbar.refresh()
    
    # 等待进程完成
    process.wait()
    
    # 确保进度条完成
    pbar.n = total_epochs
    pbar.refresh()
    pbar.close()
    
    return process.returncode

def run_training():
    models = [
        'L2SKNet_UNet',
        'L2SKNet_FPN', 
        'L2SKNet_1D_UNet',
        'L2SKNet_1D_FPN'
    ]
    
    datasets = [
        'NUDT-SIRST',
        'IRSTD-1K'
    ]
    
    total = len(models) * len(datasets)
    current = 0
    
    for dataset in datasets:
        for model in models:
            current += 1
            start_time = time.time()
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current}/{total}] {current_time} - Training {model} on {dataset}")
            
            # RTX 5090最佳配置：根据数据集选择参数
            if dataset == 'IRSTD-1K':  # 512x512
                batch_size = '3'
                num_workers = '8'
            else:  # 256x256
                batch_size = '10'
                num_workers = '2'
            
            cmd = [
                'python', 'train_device0.py',
                '--model_names', model,
                '--dataset_names', dataset,
                '--batchSize', batch_size,
                '--threads', num_workers
            ]
            
            # 使用进度条监控训练
            run_training_with_progress(cmd, model, dataset)
            
            # Calculate and print completion time
            end_time = time.time()
            duration = end_time - start_time
            completion_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current}/{total}] {completion_time} - Completed {model} on {dataset} ({duration/60:.1f} min)")

if __name__ == '__main__':
    run_training()
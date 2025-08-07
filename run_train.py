#!/usr/bin/env python3

import subprocess
import time
import re
from datetime import datetime
from tqdm import tqdm

def run_with_progress(cmd, model, dataset):
    """运行训练并显示进度"""
    pbar = tqdm(total=400, desc=f'{model} on {dataset}', unit='epoch')
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             universal_newlines=True, bufsize=1)
    
    for line in process.stdout:
        match = re.search(r'Epoch---(\d+), total_loss---([\d.]+)', line)
        if match:
            epoch, loss = int(match.group(1)), float(match.group(2))
            pbar.n = epoch
            pbar.set_postfix({'loss': f'{loss:.6f}'})
            pbar.refresh()
    
    process.wait()
    pbar.n = 400
    pbar.refresh()
    pbar.close()
    return process.returncode

def main():
    datasets = ['NUDT-SIRST']
    
    # 定义所有要训练的模型配置
    model_configs = [
        # 普通的四个网络
        ('L2SKNet_UNet', False),
        ('L2SKNet_FPN', False),
        ('L2SKNet_1D_UNet', False),
        ('L2SKNet_1D_FPN', False),
        # 两个2D带morphology的网络
        ('L2SKNet_UNet', True),
        ('L2SKNet_FPN', True),
    ]
    
    total_tasks = len(datasets) * len(model_configs)
    task_count = 0
    
    for dataset in datasets:
        for model_name, use_morphology in model_configs:
            task_count += 1
            start = time.time()
            
            model_display_name = model_name + ('_morphology' if use_morphology else '')
            print(f"[{task_count}/{total_tasks}] {datetime.now():%H:%M:%S} - Training {model_display_name} on {dataset}")
            
            cmd = ['python', 'train_device0.py', '--model_names', model_name, 
                   '--dataset_names', dataset, '--batchSize', '10', '--threads', '2']
            if use_morphology:
                cmd.append('--use_morphology')
            
            run_with_progress(cmd, model_display_name, dataset)
            
            duration = time.time() - start
            print(f"[{task_count}/{total_tasks}] {datetime.now():%H:%M:%S} - Completed {model_display_name} on {dataset} ({duration/60:.1f} min)")

if __name__ == '__main__':
    main()
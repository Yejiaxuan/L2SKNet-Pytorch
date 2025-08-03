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
    models = ['L2SKNet_UNet', 'L2SKNet_FPN', 'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN']
    datasets = ['NUDT-SIRST']
    
    for i, (dataset, model) in enumerate([(d, m) for d in datasets for m in models], 1):
        start = time.time()
        print(f"[{i}/{len(models)}] {datetime.now():%H:%M:%S} - Training {model} on {dataset}")
        
        cmd = ['python', 'train_device0.py', '--model_names', model, 
               '--dataset_names', dataset, '--batchSize', '10', '--threads', '2']
        
        run_with_progress(cmd, model, dataset)
        
        duration = time.time() - start
        print(f"[{i}/{len(models)}] {datetime.now():%H:%M:%S} - Completed {model} on {dataset} ({duration/60:.1f} min)")

if __name__ == '__main__':
    main()
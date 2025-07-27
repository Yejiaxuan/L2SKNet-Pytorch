#!/usr/bin/env python3

import subprocess
from datetime import datetime

def run_training():
    models = [
        'L2SKNet_UNet',
        'L2SKNet_FPN', 
        'L2SKNet_1D_UNet',
        'L2SKNet_1D_FPN'
    ]
    
    datasets = [
        'NUDT-SIRST',
        'IRSTD-1K', 
        'SIRST-aug'
    ]
    
    total = len(models) * len(datasets)
    current = 0
    
    for dataset in datasets:
        for model in models:
            current += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current}/{total}] {current_time} - Training {model} on {dataset}")
            
            cmd = [
                'python', 'train_device0.py',
                '--model_names', model,
                '--dataset_names', dataset
            ]
            
            subprocess.run(cmd)

if __name__ == '__main__':
    run_training()
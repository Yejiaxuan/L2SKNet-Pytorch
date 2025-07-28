#!/usr/bin/env python3

import subprocess
import time
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
            start_time = time.time()
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current}/{total}] {current_time} - Training {model} on {dataset}")
            
            # Create log file for this training
            log_file = f"training_logs/{model}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            cmd = [
                'python', 'train_device0.py',
                '--model_names', model,
                '--dataset_names', dataset
            ]
            
            # Redirect output to log file
            with open(log_file, 'w') as f:
                subprocess.run(cmd, stdout=f, stderr=f)
            
            # Calculate and print completion time
            end_time = time.time()
            duration = end_time - start_time
            completion_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current}/{total}] {completion_time} - Completed {model} on {dataset} ({duration/60:.1f} min)")

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('training_logs', exist_ok=True)
    run_training()
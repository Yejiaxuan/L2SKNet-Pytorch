#!/usr/bin/env python3
import os
import re
import subprocess
import sys

def find_best_epoch(dataset_name, model_name):
    """从日志文件中找到最佳epoch"""
    log_pattern = f"{dataset_name}_{model_name}_"
    
    for file in os.listdir("log"):
        if file.startswith(log_pattern) and file.endswith('.txt'):
            with open(f"log/{file}", 'r', encoding='utf-8') as f:
                content = f.read()
            fscore_matches = re.findall(r'Best fscore: ([\d.]+),when Epoch=(\d+)', content)
            if fscore_matches:
                _, best_epoch = fscore_matches[-1]
                return int(best_epoch)
    return 200

def find_available_epoch(dataset_name, model_name, target_epoch):
    """找到可用的epoch"""
    checkpoint_dir = f"log/{dataset_name}/{model_name}"
    epochs = [int(f.split('.')[0]) for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    
    if target_epoch in epochs:
        return target_epoch
    return min(epochs, key=lambda x: abs(x - target_epoch)) if epochs else None

def run_test(model_name, epoch):
    """运行测试"""
    subprocess.run([sys.executable, "test.py", "--model_names", model_name,
                   "--dataset_names", "NUDT-SIRST", "--test_epo", str(epoch)])

def calculate_metrics(model_name, dataset_name):
    """计算指标"""
    subprocess.run([sys.executable, "cal_metrics.py", "--model_names", model_name,
                   "--dataset_names", dataset_name])

def main():
    print("Auto Testing All Models...")
    
    models = ['L2SKNet_UNet', 'L2SKNet_FPN', 'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN']
    dataset = "NUDT-SIRST"
    
    # 准备测试计划
    test_plan = []
    for model in models:
        best_epoch = find_best_epoch(dataset, model)
        final_epoch = find_available_epoch(dataset, model, best_epoch)
        
        if final_epoch:
            test_plan.append((model, final_epoch))
            print(f"{model}: epoch {final_epoch}")
        else:
            print(f"{model}: no checkpoint found, skipped")
    
    print(f"\nTesting {len(test_plan)} models...")
    
    # 执行测试和计算指标
    for i, (model, epoch) in enumerate(test_plan, 1):
        print(f"[{i}/{len(test_plan)}] Testing {model}...")
        run_test(model, epoch)
        print(f"Calculating metrics for {model}...")
        calculate_metrics(model, dataset)
        print(f"✓ {model} completed")
    
    print("Done!")

if __name__ == '__main__':
    main()
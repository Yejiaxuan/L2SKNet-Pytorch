#!/usr/bin/env python3
import os
import re
import subprocess
import sys

def find_best_epoch(dataset_name, model_name, use_morphology=False):
    """从日志文件中找到最佳epoch"""
    model_suffix = model_name + ('_morphology' if use_morphology else '')
    log_pattern = f"{dataset_name}_{model_suffix}_"
    
    for file in os.listdir("log"):
        if file.startswith(log_pattern) and file.endswith('.txt'):
            with open(f"log/{file}", 'r', encoding='utf-8') as f:
                content = f.read()
            fscore_matches = re.findall(r'Best fscore: ([\d.]+),when Epoch=(\d+)', content)
            if fscore_matches:
                _, best_epoch = fscore_matches[-1]
                return int(best_epoch)
    return 200

def find_available_epoch(dataset_name, model_name, target_epoch, use_morphology=False):
    """找到可用的epoch"""
    model_dir = model_name + ('_morphology' if use_morphology else '')
    checkpoint_dir = f"log/{dataset_name}/{model_dir}"
    
    if not os.path.exists(checkpoint_dir):
        return None
        
    epochs = [int(f.split('.')[0]) for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    
    if target_epoch in epochs:
        return target_epoch
    return min(epochs, key=lambda x: abs(x - target_epoch)) if epochs else None

def run_test(model_name, epoch, use_morphology=False):
    """运行测试"""
    cmd = [sys.executable, "test.py", "--model_names", model_name,
           "--dataset_names", "NUDT-SIRST", "--test_epo", str(epoch)]
    if use_morphology:
        cmd.append("--use_morphology")
    subprocess.run(cmd)

def calculate_metrics(model_name, dataset_name, use_morphology=False):
    """计算指标"""
    result_model_name = model_name + ('_morphology' if use_morphology else '')
    subprocess.run([sys.executable, "cal_metrics.py", "--model_names", result_model_name,
                   "--dataset_names", dataset_name])

def main():
    print("Auto Testing All Models...")
    
    dataset = "NUDT-SIRST"
    
    # 定义所有要测试的模型配置
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
    
    # 准备测试计划
    test_plan = []
    for model_name, use_morphology in model_configs:
        best_epoch = find_best_epoch(dataset, model_name, use_morphology)
        final_epoch = find_available_epoch(dataset, model_name, best_epoch, use_morphology)
        
        model_display_name = model_name + ('_morphology' if use_morphology else '')
        if final_epoch:
            test_plan.append((model_name, final_epoch, use_morphology))
            print(f"{model_display_name}: epoch {final_epoch}")
        else:
            print(f"{model_display_name}: no checkpoint found, skipped")
    
    print(f"\nTesting {len(test_plan)} models...")
    
    # 执行测试和计算指标
    for i, (model_name, epoch, use_morphology) in enumerate(test_plan, 1):
        model_display_name = model_name + ('_morphology' if use_morphology else '')
        print(f"[{i}/{len(test_plan)}] Testing {model_display_name}...")
        run_test(model_name, epoch, use_morphology)
        print(f"Calculating metrics for {model_display_name}...")
        calculate_metrics(model_name, dataset, use_morphology)
        print(f"✓ {model_display_name} completed")
    
    print("All models evaluation completed!")

if __name__ == '__main__':
    main()
import argparse
import os
from typing import List, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net import Net
from utils.datasets import NUDTSIRSTSetLoader
from model.L2SKNet.LLSKMs import LLSKM, LLSKM_d, LLSKM_1D

# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------

def apply_colormap_on_image(org_img: np.ndarray, activation_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """将热度图叠加到原图上返回三通道 BGR 图像。"""
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    if org_img.ndim == 2:  # 灰度 -> 3 通道
        org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(heatmap, alpha, org_img, 1 - alpha, 0)
    return overlay


def find_target_layers(model: torch.nn.Module, layer_cls: Type[torch.nn.Module]) -> List[torch.nn.Module]:
    """返回网络中所有 layer_cls 实例的引用列表。"""
    return [m for m in model.modules() if isinstance(m, layer_cls)]


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLSKM 前后可视化")
    parser.add_argument("--model_name", type=str, default="L2SKNet_FPN",
                        help="模型名称: L2SKNet_UNet / L2SKNet_FPN / L2SKNet_1D_UNet / L2SKNet_1D_FPN")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重文件(.pth.tar)")
    parser.add_argument("--dataset_dir", type=str, default="./data/NUDT-SIRST/", help="NUDT-SIRST 数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./llskm_vis_output", help="输出文件夹")
    parser.add_argument("--layer_type", type=str, choices=["LLSKM", "LLSKM_d", "LLSKM_1D"], default="LLSKM",
                        help="要可视化的层类型")
    parser.add_argument("--layer_index", type=int, default=0, help="第几次出现的该层 (从0开始)")
    parser.add_argument("--max_samples", type=int, default=30, help="最大可视化样本数")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据集
    test_set = NUDTSIRSTSetLoader(base_dir=args.dataset_dir, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # 模型
    net = Net(model_name=args.model_name).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
    net.eval()

    # 选择层类别
    layer_cls_map = {"LLSKM": LLSKM, "LLSKM_d": LLSKM_d, "LLSKM_1D": LLSKM_1D}
    layer_cls = layer_cls_map[args.layer_type]
    candidate_layers = find_target_layers(net.model, layer_cls)
    if not candidate_layers:
        raise RuntimeError(f"模型中未找到层 {args.layer_type}")
    if args.layer_index >= len(candidate_layers):
        raise IndexError(f"layer_index 超出范围，可用 0~{len(candidate_layers)-1}")
    target_layer = candidate_layers[args.layer_index]
    print(f"可视化层: {target_layer} (index={args.layer_index})")

    # forward hook 捕获输入输出
    feats_in, feats_out = {}, {}

    def hook(module, inp, out):
        feats_in["feat"] = inp[0].detach().cpu()
        feats_out["feat"] = out.detach().cpu()

    handle = target_layer.register_forward_hook(hook)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (img_tensor, _, size, name) in enumerate(test_loader):
            if idx >= args.max_samples:
                break
            img_tensor = img_tensor.to(device)
            feats_in.clear(); feats_out.clear()
            _ = net(img_tensor)  # 前向推理，同时触发 hook

            if not feats_in:
                print("Warning: 未捕获到特征，跳过")
                continue

            x_in = feats_in["feat"][0]   # (C,H,W)
            x_out = feats_out["feat"][0]  # (C,H,W)

            # 通道平均 + ReLU
            def tensor_to_map(x: torch.Tensor):
                m = F.relu(x).mean(0)  # (H,W)
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                return m.numpy()

            map_in = tensor_to_map(x_in)
            map_out = tensor_to_map(x_out)

            h, w = size[0].item(), size[1].item()
            map_in = cv2.resize(map_in, (w, h))
            map_out = cv2.resize(map_out, (w, h))

            # 原图
            img_path = os.path.join(args.dataset_dir, "images", name[0] + ".png")
            org_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if org_img is None:
                print(f"Warning: 无法读取 {img_path}")
                continue

            before_vis = apply_colormap_on_image(org_img, map_in, alpha=0.4)
            after_vis = apply_colormap_on_image(org_img, map_out, alpha=0.4)

            cv2.imwrite(os.path.join(args.output_dir, f"{name[0]}_before.png"), before_vis)
            cv2.imwrite(os.path.join(args.output_dir, f"{name[0]}_after.png"), after_vis)
            print(f"Saved {name[0]} before/after")

    handle.remove()
    print("LLSKM 可视化完成！")


if __name__ == "__main__":
    main()
# coding=utf-8
"""
FoveatedViT 验证脚本
运行此脚本确认环境配置正确
"""

import sys
import os
print("Python version:", sys.version)

# 1. 测试基础库
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import numpy as np
print(f"NumPy version: {np.__version__}")

# 2. 测试模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.foveated_vit import FoveatedViT, get_foveated_b16_small_config

config = get_foveated_b16_small_config()
model = FoveatedViT(config=config, img_size=224, num_classes=10, zero_head=True)
print(f"\nModel created successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 3. 测试前向传播
x = torch.randn(2, 3, 224, 224)
logits, _ = model(x)
print(f"Forward pass: input {x.shape} -> output {logits.shape}")

# 4. 测试训练循环
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for step in range(2):
    loss = model(x, labels=torch.randint(0, 10, (2,)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step+1}: loss={loss.item():.4f}")

print("\nAll tests passed! Environment is ready.")

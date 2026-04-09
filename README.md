# FocViT

基于人类视觉 foveal 机制的重铸视觉 Transformer，通过中心高分辨率、外围低分辨率的设计实现 token 数量削减。

**FocViT** = **Foc**eated **Vi**sion **T**ransformer

## 核心思想

人类视觉具有 **foveal 机制**：
- **中心凹**（fovea）：视野中心区域，分辨率最高
- **外围**：视野边缘区域，分辨率逐渐降低

FocViT 将此机制引入 Vision Transformer：
- **中心区域**（r ≤ 10.5）：8×8 patch，高分辨率
- **外围区域**（10.5 < r ≤ 13.5）：2×2 block 合并为 16×16，降低分辨率
- **丢弃区域**（r > 13.5）：四角 patch 被丢弃

```
标准 ViT:  784 tokens (28×28)
FocViT: 456 tokens (减少 41.8%)
```

## 项目结构

```
FocViT/
├── models/
│   ├── foveated_vit.py     # 核心模型实现
│   ├── configs.py           # 配置文件
│   ├── modeling.py          # 标准 ViT (参考)
│   └── modeling_resnet.py  # ResNet 混合模型
├── train_foveated.py        # 训练脚本 (OCT-MNIST)
├── example_foveated.py      # 使用示例
├── visualize_foveated.py    # 可视化工具
├── count_params.py          # 参数量计算
├── requirements.txt         # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
python example_foveated.py
```

输出示例：
```
Token Information:
  Center tokens (8x8 patches, r <= 10.5): 332
  Outer tokens (16x16 merged): 124
  Total tokens (excl. cls): 456
  Token reduction: 41.8%
```

### 3. 训练模型

```bash
python train_foveated.py --batch_size 32 --num_epochs 100
```

参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 64 | 批量大小 |
| `--num_epochs` | 100 | 训练轮数 |
| `--lr` | 1e-5 | 学习率 |
| `--data_dir` | data/octmnist_jpg_subset | 数据目录 |

### 4. 生成可视化

```bash
python visualize_foveated.py
```

生成图像：
- `myplot.png`：foveated 区域划分
- `comparison_plot.png`：标准 ViT vs FocViT 对比

## 模型配置

| 配置 | hidden_size | num_layers | num_heads | 参数量 |
|------|-------------|------------|-----------|--------|
| **FocViT-B/8** | 768 | 12 | 12 | ~86M |
| **FocViT-Small** | 384 | 6 | 6 | ~22M |

### 可用配置函数

```python
from models.foveated_vit import (
    get_foveated_b16_full_config,   # ViT-B/16 容量
    get_foveated_b16_small_config,  # 小型配置
    get_foveated_b16_config         # 基础配置
)
```

## Token 分配详解

对于 224×224 图像，8×8 base patch：

```
Grid: 28×28 = 784 patches

┌─────────────────────────────┐
│          discarded           │
│     ┌───────────────┐        │
│     │   outer (r)   │        │
│     │ ┌───────────┐ │        │
│     │ │  center   │ │        │
│     │ │  (r≤10.5) │ │        │
│     │ └───────────┘ │        │
│     └───────────────┘        │
│          discarded           │
└─────────────────────────────┘

- Center (8×8 patches): 332 tokens
- Outer (16×16 merged): 124 tokens  
- Total: 456 tokens (+ 1 cls = 457)
```

## 与标准 ViT 对比

| 指标 | 标准 ViT-B/16 | FocViT-B/8 | 变化 |
|------|---------------|------------------|------|
| Patch 大小 | 16×16 | 8×8 / 16×16 混合 | - |
| Token 数量 | 197 (含 cls) | 457 (含 cls) | - |
| Token 削减率 | - | 41.8% | ✓ |
| 参数量 | ~86M | ~86M | 相当 |
| 中心分辨率 | 相同 | 更高 (8×8 vs 16×16) | ✓ |

## 依赖

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
ml-collections>=0.1.0
tqdm>=4.62.0
scipy>=1.7.0
matplotlib>=3.3.0
Pillow>=8.0.0
```

## 致谢

参考实现：
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [DeiT](https://github.com/facebookresearch/deit)

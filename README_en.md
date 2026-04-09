# FocViT

A foveated Vision Transformer that mimics the human visual foveal mechanism, achieving token reduction through high-resolution center and low-resolution periphery design.

**FocViT** = **Foc**eated **Vi**sion **T**ransformer

## Core Idea

Human vision has a **foveal mechanism**:
- **Fovea**: Central vision with highest resolution
- **Periphery**: Edge of vision with gradually decreasing resolution

FocViT brings this mechanism into Vision Transformer:
- **Center region** (r ≤ 10.5): 8×8 patches, high resolution
- **Outer region** (10.5 < r ≤ 13.5): 2×2 blocks merged to 16×16, lower resolution
- **Discarded region** (r > 13.5): Corner patches are discarded

```
Standard ViT:  784 tokens (28×28)
FocViT: 456 tokens (41.8% reduction)
```

## Project Structure

```
FocViT/
├── models/
│   ├── foveated_vit.py     # Core model implementation
│   ├── configs.py           # Configuration files
│   ├── modeling.py          # Standard ViT (reference)
│   └── modeling_resnet.py  # ResNet hybrid model
├── train_foveated.py        # Training script (OCT-MNIST)
├── example_foveated.py      # Usage examples
├── visualize_foveated.py    # Visualization tools
├── count_params.py          # Parameter counting
├── requirements.txt         # Dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Example

```bash
python example_foveated.py
```

Output example:
```
Token Information:
  Center tokens (8x8 patches, r <= 10.5): 332
  Outer tokens (16x16 merged): 124
  Total tokens (excl. cls): 456
  Token reduction: 41.8%
```

### 3. Train Model

```bash
python train_foveated.py --batch_size 32 --num_epochs 100
```

Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 64 | Batch size |
| `--num_epochs` | 100 | Number of epochs |
| `--lr` | 1e-5 | Learning rate |
| `--data_dir` | data/octmnist_jpg_subset | Data directory |

### 4. Generate Visualization

```bash
python visualize_foveated.py
```

Generates:
- `myplot.png`: Foveated region division
- `comparison_plot.png`: Standard ViT vs FocViT comparison

## Model Configurations

| Config | hidden_size | num_layers | num_heads | Parameters |
|--------|-------------|------------|-----------|------------|
| **FocViT-B/8** | 768 | 12 | 12 | ~86M |
| **FocViT-Small** | 384 | 6 | 6 | ~22M |

### Available Configuration Functions

```python
from models.foveated_vit import (
    get_foveated_b16_full_config,   # ViT-B/16 capacity
    get_foveated_b16_small_config,  # Small config
    get_foveated_b16_config         # Base config
)
```

## Token Distribution Details

For 224×224 images with 8×8 base patches:

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

## Comparison with Standard ViT

| Metric | Standard ViT-B/16 | FocViT-B/8 | Change |
|--------|-------------------|-----------------|--------|
| Patch size | 16×16 | 8×8 / 16×16 hybrid | - |
| Token count | 197 (with cls) | 457 (with cls) | - |
| Token reduction | - | 41.8% | ✓ |
| Parameters | ~86M | ~86M | Similar |
| Center resolution | Same | Higher (8×8 vs 16×16) | ✓ |

## Dependencies

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

## Acknowledgements

Reference implementations:
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [DeiT](https://github.com/facebookresearch/deit)

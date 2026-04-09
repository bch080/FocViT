# coding=utf-8
"""精确计算 FoveatedViT 参数量"""

import sys
import os

# 确保可以导入 models 模块
current_file = os.path.dirname(os.path.abspath(__file__))
if current_file not in sys.path:
    sys.path.insert(0, current_file)

import math
from models.foveated_vit import FoveatedViT, get_foveated_b16_full_config, get_foveated_b16_small_config


def compute_foveated_tokens():
    """计算 foveated patch 的 token 数量

    当前实现：
    - grid_size = 28 (224 / 8)
    - center = 13.5
    - R_keep = 13.0
    - R_center = 9.5
    - Center: r <= 9.5, 使用 8x8 patch
    - Outer: 9.5 < r <= 13.0, 使用 8x8 patch
    """
    grid_size = 28  # 224 / 8
    center = 13.5
    R_keep = 13.0
    R_center = 9.5

    center_patches = []
    outer_patches = []
    discarded_patches = []

    for i in range(grid_size):
        for j in range(grid_size):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)

            if dist <= R_center:
                center_patches.append((i, j))
            elif dist <= R_keep:
                outer_patches.append((i, j))
            else:
                discarded_patches.append((i, j))

    return len(center_patches), len(outer_patches), len(discarded_patches)


num_center, num_outer, num_discarded = compute_foveated_tokens()
num_patches = num_center + num_outer

print("=" * 70)
print("FoveatedViT 参数量精确计算")
print("=" * 70)

print(f"\n【Token 信息】")
print(f"  中心 tokens (r <= 9.5): {num_center}")
print(f"  外围 tokens (9.5 < r <= 13.0): {num_outer}")
print(f"  总 tokens: {num_patches}")
print(f"  丢弃 tokens (r > 13.0): {num_discarded}")
print(f"  CLS token: 1")
print(f"  序列长度: {num_patches + 1}")
print(f"  Token 减少: {(1 - num_patches / 784) * 100:.1f}%")


# ============ 配置 1: B/16 容量 ============
print(f"\n{'='*70}")
print("【配置 1: FoveatedViT-B/8 (ViT-B/16 容量)】")
print(f"{'='*70}")

config_full = get_foveated_b16_full_config()
print(f"\n配置参数:")
print(f"  hidden_size: {config_full.hidden_size}")
print(f"  num_heads: {config_full.transformer.num_heads}")
print(f"  num_layers: {config_full.transformer.num_layers}")
print(f"  mlp_dim: {config_full.transformer.mlp_dim}")

D = config_full.hidden_size  # 768
H = config_full.transformer.num_heads  # 12
L = config_full.transformer.num_layers  # 12
M = config_full.transformer.mlp_dim  # 3072

print(f"\n{'='*70}")
print("【FoveatedViT-B/8 参数量分解】")
print(f"{'='*70}")

# 1. FoveatedPatchEmbed
print(f"\n1. FoveatedPatchEmbed")
# cls_token
cls_params = D
print(f"   cls_token: 1 × {D} = {cls_params:,}")

# center_proj: 8*8*3 = 192 -> D (center 和 outer 都用这个)
center_in = 8 * 8 * 3  # 192
proj_params = center_in * D + D
print(f"   center_proj: Linear({center_in}, {D})")
print(f"     {center_in} × {D} + {D} = {proj_params:,}")

# dropout
dropout_params = 0  # Dropout 没有参数

patch_embed_total = cls_params + proj_params
print(f"   小计: {patch_embed_total:,}")

# 2. Position Embeddings
pos_embed_params = (num_patches + 1) * D
print(f"\n2. Position Embeddings")
print(f"   ({num_patches} + 1) × {D} = {pos_embed_params:,}")

# 3. Encoder
print(f"\n3. Transformer Encoder")

# Attention per layer: Q, K, V, O 四个 linear
qkv_params = 3 * D * D  # Q, K, V
o_params = D * D  # O
attention_per_layer = qkv_params + o_params
print(f"   Attention 单层:")
print(f"     Q, K, V: 3 × {D} × {D} = {qkv_params:,}")
print(f"     O: {D} × {D} = {o_params:,}")
print(f"     小计: {attention_per_layer:,}")

# MLP per layer
fc1_params = D * M
fc2_params = M * D
mlp_per_layer = fc1_params + fc2_params
print(f"   MLP 单层:")
print(f"     FC1: {D} × {M} = {fc1_params:,}")
print(f"     FC2: {M} × {D} = {fc2_params:,}")
print(f"     小计: {mlp_per_layer:,}")

# LayerNorm per layer
ln_params = 2 * D
print(f"   LayerNorm 单层: 2 × {D} = {ln_params:,}")

block_per_layer = attention_per_layer + mlp_per_layer + 2 * ln_params
print(f"   Block 单层: {block_per_layer:,}")

encoder_blocks = L * block_per_layer
print(f"   Encoder Blocks: {L} × {block_per_layer:,} = {encoder_blocks:,}")

encoder_norm = 2 * D
print(f"   Encoder Norm: {encoder_norm:,}")

encoder_total = encoder_blocks + encoder_norm
print(f"   Encoder 总计: {encoder_total:,}")

# 4. Head
head_params = D * 10 + 10
print(f"\n4. Classification Head")
print(f"   Linear({D}, 10): {D} × 10 + 10 = {head_params:,}")

# 总计
total_full = patch_embed_total + pos_embed_params + encoder_total + head_params
print(f"\n{'='*70}")
print(f"【总计】FoveatedViT-B/8: {total_full:,} = {total_full/1e6:.2f}M")
print(f"{'='*70}")

# ============ 配置 2: Small ============
print(f"\n\n{'='*70}")
print("【配置 2: FoveatedViT-Small】")
print(f"{'='*70}")

config_small = get_foveated_b16_small_config()
print(f"\n配置参数:")
print(f"  hidden_size: {config_small.hidden_size}")
print(f"  num_heads: {config_small.transformer.num_heads}")
print(f"  num_layers: {config_small.transformer.num_layers}")
print(f"  mlp_dim: {config_small.transformer.mlp_dim}")

D_s = config_small.hidden_size  # 384
M_s = config_small.transformer.mlp_dim  # 1536
L_s = config_small.transformer.num_layers  # 6

# Patch Embed
cls_s = D_s
proj_s = (8 * 8 * 3) * D_s + D_s
patch_embed_s = cls_s + proj_s

# Pos Embed
pos_embed_s = (num_patches + 1) * D_s

# Encoder
qkv_s = 3 * D_s * D_s
o_s = D_s * D_s
att_s = qkv_s + o_s
fc1_s = D_s * M_s
fc2_s = M_s * D_s
mlp_s = fc1_s + fc2_s
ln_s = 2 * D_s
block_s = att_s + mlp_s + 2 * ln_s
enc_s = L_s * block_s + 2 * D_s

# Head
head_s = D_s * 10 + 10

total_small = patch_embed_s + pos_embed_s + enc_s + head_s
print(f"\n【总计】FoveatedViT-Small: {total_small:,} = {total_small/1e6:.2f}M")

# ============ 标准 ViT-B/16 对比 ============
print(f"\n\n{'='*70}")
print("【标准 ViT-B/16 对比】")
print(f"{'='*70}")

std_D = 768
std_L = 12
std_M = 3072
std_patches = 14 * 14  # 196

# Patch Embed (Conv2d 16x16)
std_patch_embed = 3 * 16 * 16 * std_D + std_D
print(f"\n标准 ViT-B/16:")
print(f"  Patch Embed (Conv2d): 3×16×16×{std_D}+{std_D} = {std_patch_embed:,}")

# Pos Embed
std_pos = (std_patches + 1) * std_D
print(f"  Position Embed: ({std_patches}+1)×{std_D} = {std_pos:,}")

# Encoder
std_qkv = 3 * std_D * std_D
std_o = std_D * std_D
std_att = std_qkv + std_o
std_fc1 = std_D * std_M
std_fc2 = std_M * std_D
std_mlp = std_fc1 + std_fc2
std_ln = 2 * std_D
std_block = std_att + std_mlp + 2 * std_ln
std_enc = std_L * std_block + 2 * std_D
print(f"  Encoder: {std_L}×{std_block:,}+{2*std_D} = {std_enc:,}")

# Head
std_head = std_D * 1000 + 1000
print(f"  Head (1000 classes): {std_head:,}")

std_total = std_patch_embed + std_pos + std_enc + std_head
print(f"\n  总计: {std_total:,} = {std_total/1e6:.2f}M")

# ============ 总结 ============
print(f"\n\n{'='*70}")
print("【三模型参数量对比】")
print(f"{'='*70}")
print(f"\n  {'模型':<30} {'参数量':>15} {'Token数':>10}")
print(f"  {'-'*30} {'-'*15} {'-'*10}")
print(f"  {'标准 ViT-B/16':<30} {std_total/1e6:>14.2f}M {std_patches+1:>10}")
print(f"  {'FoveatedViT-B/8':<30} {total_full/1e6:>14.2f}M {num_patches+1:>10}")
print(f"  {'FoveatedViT-Small':<30} {total_small/1e6:>14.2f}M {num_patches+1:>10}")
print(f"  {'-'*30} {'-'*15} {'-'*10}")
print(f"  {'FoveatedViT-B/8 vs ViT-B/16 差距':<30} {(total_full-std_total)/1e6:>+14.2f}M")
print(f"  {'相对差距':<30} {(total_full-std_total)/std_total*100:>+14.2f}%")

# ============ PyTorch 验证 ============
print(f"\n{'='*70}")
print("【PyTorch 实际验证】")
print(f"{'='*70}")

model_full = FoveatedViT(config=config_full, img_size=224, num_classes=10)
actual_full = sum(p.numel() for p in model_full.parameters())
print(f"  FoveatedViT-B/8 实际: {actual_full:,} ({actual_full/1e6:.2f}M)")
print(f"  计算值: {total_full:,} ({total_full/1e6:.2f}M)")
print(f"  验证: {'✓ 通过' if actual_full == total_full else '✗ 不匹配'}")

model_small = FoveatedViT(config=config_small, img_size=224, num_classes=10)
actual_small = sum(p.numel() for p in model_small.parameters())
print(f"\n  FoveatedViT-Small 实际: {actual_small:,} ({actual_small/1e6:.2f}M)")
print(f"  计算值: {total_small:,} ({total_small/1e6:.2f}M)")
print(f"  验证: {'✓ 通过' if actual_small == total_small else '✗ 不匹配'}")
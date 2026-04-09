# coding=utf-8
"""打印 FoveatedViT 的所有参数和超参数（不依赖外部库）"""

import math


def main():
    print("=" * 60)
    print("  FoveatedViT 参数与超参数")
    print("=" * 60)

    # ==================== 硬编码配置 ====================
    configs = {
        'FoveatedViT-B/8 (Full)': {
            'hidden_size': 768,
            'num_heads': 12,
            'num_layers': 12,
            'mlp_dim': 3072,
            'attention_dropout': 0.0,
            'dropout': 0.1,
            'num_classes': 4,
        },
        'FoveatedViT-Small': {
            'hidden_size': 384,
            'num_heads': 6,
            'num_layers': 6,
            'mlp_dim': 1536,
            'attention_dropout': 0.0,
            'dropout': 0.1,
            'num_classes': 4,
        },
    }

    # ==================== Foveated 视觉参数 ====================
    print(f"\n{'='*60}")
    print(f"Foveated 视觉参数 (所有配置共用)")
    print(f"{'='*60}")
    print(f"""
  img_size:         224
  base_patch_size:   8
  grid_size:         28 (= 224 / 8)
  num_base_patches: 784 (= 28 x 28)

  center:          13.5  (圆心坐标)
  R_keep:          13.5  (保留半径)
  R_center:        10.5  (中心区域半径)

  区域划分:
    - 中心区域:    r <= 10.5, 使用 8x8 patch
    - 外围区域: 10.5 < r <= 13.5, 使用 16x16 patch (2x2合并)
    - 丢弃区域:    r > 13.5, 不使用
    """)

    # ==================== 计算 token 数量 ====================
    grid_size = 28
    center = 13.5
    R_keep = 13.5
    R_center = 10.5

    # 中心 patches: r <= R_center 的所有 8x8 patches
    center_patches = []
    discarded_patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist <= R_center:
                center_patches.append((i, j))
            elif dist > R_keep:
                discarded_patches.append((i, j))

    # 外围 patches: 2x2 blocks 的左上角
    outer_patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            if i % 2 == 0 and j % 2 == 0:
                block_valid = True
                for di in range(2):
                    for dj in range(2):
                        ni, nj = i + di, j + dj
                        if ni >= grid_size or nj >= grid_size:
                            block_valid = False
                            break
                        block_dist = math.sqrt((ni - center) ** 2 + (nj - center) ** 2)
                        if block_dist > R_keep:
                            block_valid = False
                            break
                    if not block_valid:
                        break
                if block_valid:
                    outer_patches.append((i, j))

    center_tokens = len(center_patches)
    outer_tokens = len(outer_patches)
    total_tokens = center_tokens + outer_tokens
    discarded = len(discarded_patches)

    print(f"\n{'='*60}")
    print(f"Token 信息")
    print(f"{'='*60}")
    print(f"""
  Token 统计:
    Center tokens (8x8):   {center_tokens:>4} (r <= 10.5)
    Outer tokens (16x16):    {outer_tokens:>4} (10.5 < r <= 13.5)
    Total tokens:          {total_tokens:>4} (不含 CLS)
    CLS token:                    1
    序列长度:              {total_tokens + 1:>4} (含 CLS)
    Discarded:             {discarded:>4} (r > 13.5)

  Token 减少对比:
    标准 ViT-B/16:  196 tokens (14x14)
    当前模型:       {total_tokens} tokens
    减少比例:       {(1 - total_tokens / 196) * 100:.1f}%

  外围区域覆盖:
    16x16 blocks:   {outer_tokens} 个
    覆盖 base patches: {outer_tokens * 4} 个
    覆盖率:        {outer_tokens * 4 / 784 * 100:.1f}% of image area
    """)

    # ==================== 打印各配置详情 ====================
    for config_name, cfg in configs.items():
        D = cfg['hidden_size']
        H = cfg['num_heads']
        L = cfg['num_layers']
        M = cfg['mlp_dim']

        print(f"\n{'='*60}")
        print(f"配置: {config_name}")
        print(f"{'='*60}")

        print(f"\n【模型架构超参数】")
        print(f"  hidden_size:           {D}")
        print(f"  patch_size:            (8, 8) base, 中心8x8 / 外围16x16")
        print(f"  num_heads:             {H}")
        print(f"  num_layers:            {L}")
        print(f"  mlp_dim:              {M} (ratio = {M/D:.1f})")
        print(f"  attention_dropout_rate: {cfg['attention_dropout']}")
        print(f"  dropout_rate:           {cfg['dropout']}")
        print(f"  num_classes:           {cfg['num_classes']}")

        # ==================== 计算参数量 ====================
        print(f"\n【参数量分解】")

        # 1. Patch Embeddings
        cls_token = D
        center_proj = (8 * 8 * 3) * D + D  # 8*8*3 = 192
        outer_proj = (16 * 16 * 3) * D + D  # 16*16*3 = 768
        patch_embed_total = cls_token + center_proj + outer_proj
        print(f"  Patch Embeddings:")
        print(f"    cls_token:           {cls_token:>12,}  (1 x {D})")
        print(f"    center_proj:        {center_proj:>12,}  (192 x {D} + {D})")
        print(f"    outer_proj:         {outer_proj:>12,}  (768 x {D} + {D})")
        print(f"    小计:               {patch_embed_total:>12,}")

        # 2. Position Embeddings
        pos_embed = (total_tokens + 1) * D
        print(f"  Position Embeddings:")
        print(f"    ({total_tokens}+1) x {D} = {pos_embed:>10,}")

        # 3. Transformer Encoder
        print(f"  Transformer Encoder ({L} layers):")

        # Attention
        qkv = 3 * D * D
        o = D * D
        attn = qkv + o
        print(f"    Attention 单层:")
        print(f"      Q, K, V:        {qkv:>12,}  (3 x {D} x {D})")
        print(f"      O:               {o:>12,}  ({D} x {D})")
        print(f"      小计:           {attn:>12,}")

        # MLP
        fc1 = D * M
        fc2 = M * D
        mlp = fc1 + fc2
        print(f"    MLP 单层:")
        print(f"      FC1:             {fc1:>12,}  ({D} x {M})")
        print(f"      FC2:             {fc2:>12,}  ({M} x {D})")
        print(f"      小计:           {mlp:>12,}")

        # LayerNorm
        ln = 2 * D
        print(f"    LayerNorm 单层:    {ln:>12,}  (2 x {D})")

        # Block
        block = attn + mlp + 2 * ln
        print(f"    Block 单层:        {block:>12,}")

        # Encoder total
        encoder = L * block + 2 * D
        print(f"    Encoder ({L}层):  {encoder:>12,}")

        # 4. Head
        head = D * cfg['num_classes'] + cfg['num_classes']
        print(f"  Classification Head:")
        print(f"    {D} x {cfg['num_classes']} + {cfg['num_classes']} = {head:>10,}")

        # Total
        total = patch_embed_total + pos_embed + encoder + head
        print(f"\n  {'='*40}")
        print(f"  总参数量:         {total:>12,} = {total/1e6:.2f}M")
        print(f"  {'='*40}")

    # ==================== 对比表格 ====================
    print(f"\n\n{'='*60}")
    print("模型对比")
    print(f"{'='*60}")

    print(f"""
  {'模型':<28} {'参数量':>12} {'Tokens':>8} {'备注':<25}
  {'-'*28} {'-'*12} {'-'*8} {'-'*25}
  {'标准 ViT-B/16':<28} {'~86M':>12} {'197':>8} {'16x16 patches':<25}
  {'标准 ViT-B/8':<28} {'~86M':>12} {'785':>8} {'8x8 patches':<25}
  {'FoveatedViT-B/8 (Full)':<28} {'~86M':>12} {total_tokens+1:>8} {'8x8+16x16 foveated':<25}
  {'FoveatedViT-Small':<28} {'~16M':>12} {total_tokens+1:>8} {'8x8+16x16 foveated':<25}
    """)

    # ==================== 训练超参数 ====================
    print(f"\n{'='*60}")
    print("训练超参数推荐 (train_foveated.py)")
    print(f"{'='*60}")
    print(f"""
  数据集:    OCT-MNIST (224x224, 4类)
  图像预处理:
    - RandomHorizontalFlip
    - ToTensor
    - Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

  训练参数:
    --data_dir     data/octmnist_jpg_subset
    --output_dir   output
    --batch_size   64          (根据显存调整)
    --num_epochs   100
    --lr           1e-5        (AdamW)
    --weight_decay 0.05
    --num_workers  4
    --seed         42

  优化器:    AdamW(lr=1e-5, weight_decay=0.05)
  调度器:    CosineAnnealingLR(T_max=100, eta_min=1e-6)
  损失函数:  CrossEntropyLoss
    """)

    # ==================== FLOPs 估算 ====================
    print(f"\n{'='*60}")
    print("FLOPs 估算 (单次前向传播)")
    print(f"{'='*60}")

    D = 768
    L = 12
    M = 3072
    n = total_tokens + 1  # 序列长度

    # Attention FLOPs: QKV + attention scores
    attn_flops = 4 * n * n * D * L
    # MLP FLOPs
    mlp_flops = 2 * n * D * M * L

    total_flops = attn_flops + mlp_flops

    print(f"""
  序列长度:       {n} (含 CLS)
  Hidden size:    {D}
  Num layers:     {L}
  MLP dim:        {M}

  Attention FLOPs: {attn_flops:>15,} = {attn_flops/1e9:.2f}G
  MLP FLOPs:       {mlp_flops:>15,} = {mlp_flops/1e9:.2f}G
  总计:           {total_flops:>15,} = {total_flops/1e9:.2f}G

  对比:
    ViT-B/16 (196 tokens): ~17.6G
    ViT-B/8 (785 tokens): ~281G
    FoveatedViT ({total_tokens} tokens): ~{total_flops/1e9:.1f}G
    """)


if __name__ == "__main__":
    main()

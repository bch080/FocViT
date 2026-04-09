# coding=utf-8
"""
FoveatedViT Usage Example

This script demonstrates:
1. How to create and use the FoveatedViT model
2. Forward pass through the model
3. Token statistics and region division
4. Comparison with standard ViT
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.foveated_vit import FoveatedViT, get_foveated_b16_config, get_foveated_b16_small_config


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def example_basic_usage():
    """Basic usage example of FoveatedViT."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create model
    config = get_foveated_b16_small_config()
    model = FoveatedViT(
        config=config,
        img_size=224,
        num_classes=10,  # CIFAR-10
        zero_head=True,
        vis=False,
        base_patch_size=8
    )

    # Get token information
    token_info = model.get_token_info()
    print(f"\nToken Information:")
    print(f"  Center tokens (8x8 patches, r <= 10.5): {token_info['num_center_tokens']}")
    print(f"  Outer tokens (16x16 merged): {token_info['num_outer_tokens']}")
    print(f"  Total tokens (excl. cls): {token_info['num_total_tokens']}")
    print(f"  Total tokens (incl. cls): {token_info['num_total_tokens'] + 1}")
    print(f"  Discarded patches: {len(token_info['discarded_patches'])}")
    print(f"  Standard ViT tokens: 784")
    reduction = (1 - token_info['num_total_tokens'] / 784) * 100
    print(f"  Token reduction: {reduction:.1f}%")

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass (training mode)
    print(f"\nForward pass (training mode):")
    output = model(dummy_input, labels=torch.randint(0, 10, (batch_size,)))
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Loss: {output.item():.4f}")

    # Forward pass (evaluation mode)
    print(f"\nForward pass (eval mode):")
    logits, attn_weights = model(dummy_input)
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Attention weights: {len(attn_weights)} layers")

    # Model size
    num_params = count_parameters(model)
    print(f"\nModel size: {num_params:.2f}M parameters")


def example_token_statistics():
    """Detailed token statistics."""
    print("\n" + "=" * 60)
    print("Example 2: Token Statistics")
    print("=" * 60)

    config = get_foveated_b16_small_config()
    model = FoveatedViT(config=config, img_size=224, num_classes=10)

    token_info = model.get_token_info()

    print(f"\nGrid size: {token_info['grid_size']}x{token_info['grid_size']}")
    print(f"Image size: 224x224")
    print(f"Base patch size: 8x8")

    print(f"\nRegion Division:")
    print(f"  Center region (r <= 10.5):")
    print(f"    - {token_info['num_center_tokens']} patches")
    print(f"    - Each patch: 8x8 pixels")
    print(f"    - Each token: 8*8*3 = 192 values")

    print(f"\n  Outer region (10.5 < r <= 13.5):")
    print(f"    - {token_info['num_outer_tokens']} merged patches (16x16)")
    print(f"    - Each patch: 16x16 pixels")
    print(f"    - Each token: 16*16*3 = 768 values (from 4x 8x8 patches)")

    print(f"\n  Discarded region (r > 13.5):")
    print(f"    - {len(token_info['discarded_patches'])} patches")
    print(f"    - Corner patches are discarded")

    # Show some example patches
    print(f"\nExample center patches (first 5):")
    for i, (row, col) in enumerate(token_info['center_patches'][:5]):
        print(f"  [{i}] Row {row}, Col {col}")

    print(f"\nExample outer patches (first 5):")
    for i, (row, col) in enumerate(token_info['outer_patches'][:5]):
        print(f"  [{i}] Top-left at Row {row}, Col {col} (covers 2x2 block: {row}:{row+2}, {col}:{col+2})")


def example_forward_comparison():
    """Compare forward pass between standard ViT and FoveatedViT."""
    print("\n" + "=" * 60)
    print("Example 3: Forward Pass Comparison")
    print("=" * 60)

    from models.modeling import VisionTransformer, CONFIGS

    # Standard ViT (B/16 with 16x16 patches)
    standard_config = CONFIGS['ViT-B_16']
    standard_model = VisionTransformer(
        config=standard_config,
        img_size=224,
        num_classes=10
    )
    standard_params = count_parameters(standard_model)
    standard_tokens = 196  # (224/16)^2 = 14*14 = 196 + 1 cls = 197

    # Foveated ViT (B/8 with foveated 8x8/16x16 patches)
    foveated_config = get_foveated_b16_config()
    foveated_model = FoveatedViT(
        config=foveated_config,
        img_size=224,
        num_classes=10
    )
    foveated_params = count_parameters(foveated_model)
    foveated_tokens = foveated_model.get_token_info()['num_total_tokens'] + 1

    print(f"\n{'Model':<20} {'Parameters':<15} {'Tokens':<10} {'Token Reduction'}")
    print(f"{'-' * 60}")
    print(f"{'Standard ViT-B/16':<20} {standard_params:<15.2f}M {standard_tokens:<10} {'-'}")
    print(f"{'FoveatedViT-B/8':<20} {foveated_params:<15.2f}M {foveated_tokens:<10} {(1 - foveated_tokens/standard_tokens)*100:.1f}%")

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"\nForward pass test (batch_size={batch_size}):")

    with torch.no_grad():
        # Standard ViT
        standard_logits, _ = standard_model(dummy_input)
        print(f"  Standard ViT output: {standard_logits.shape}")

        # Foveated ViT
        foveated_logits, _ = foveated_model(dummy_input)
        print(f"  FoveatedViT output: {foveated_logits.shape}")

    print("\nBoth models produce the same output shape for classification!")


def example_training_loop():
    """Example of a simple training loop."""
    print("\n" + "=" * 60)
    print("Example 4: Simple Training Loop")
    print("=" * 60)

    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Create model
    config = get_foveated_b16_small_config()
    model = FoveatedViT(
        config=config,
        img_size=224,
        num_classes=10,
        zero_head=True
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nOptimizer: AdamW (lr=3e-4, weight_decay=0.05)")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Loss: CrossEntropyLoss")

    # Simulate a few training steps
    print("\nSimulating training steps...")
    model.train()

    for step in range(5):
        # Dummy batch
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 10, (8,))

        # Forward
        loss = model(images, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    scheduler.step()
    print(f"\nAfter scheduler step: LR = {scheduler.get_last_lr()[0]:.6f}")


def generate_visualization():
    """Generate the region division visualization."""
    print("\n" + "=" * 60)
    print("Generating Visualization: myplot.png")
    print("=" * 60)

    try:
        import visualize_foveated
        visualize_foveated.visualize_foveated_vision('myplot.png')
        print("\nVisualization saved to: myplot.png")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("You can run 'python visualize_foveated.py' separately.")


def main():
    print("\n" + "=" * 60)
    print("  FoveatedViT Usage Examples")
    print("  Foveated Vision Transformer for CIFAR-10")
    print("=" * 60)

    # Run examples
    example_basic_usage()
    example_token_statistics()
    example_forward_comparison()
    example_training_loop()

    # Generate visualization
    generate_visualization()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

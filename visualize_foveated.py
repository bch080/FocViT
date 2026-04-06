# coding=utf-8
"""
Foveated Vision Visualization

This script generates a visualization of the foveated vision region division:
- Center region: 8x8 patches (red/blue)
- Outer region: 16x16 merged patches (yellow/green)
- Discarded region: gray

The visualization shows:
1. 28x28 grid representing patches
2. Circular boundaries at r=10.5 and r=13.5
3. Color-coded regions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math


def compute_foveated_regions():
    """
    Compute which patches belong to center, outer, or discarded regions.

    Returns:
        center_patches: List of (i, j) positions
        outer_patches: List of (i, j) positions (top-left of 16x16 blocks)
        discarded_patches: List of (i, j) positions
    """
    grid_size = 28
    center = 13.5
    R_keep = 13.5
    R_center = 10.5

    center_patches = []
    outer_patches = []
    discarded_patches = []

    for i in range(grid_size):
        for j in range(grid_size):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)

            if dist <= R_keep:
                if dist <= R_center:
                    center_patches.append((i, j))
                else:
                    # Check if this is the top-left of a valid 2x2 block
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
            else:
                discarded_patches.append((i, j))

    return center_patches, outer_patches, discarded_patches


def get_outer_patch_coverage(outer_patches):
    """
    Get all base patches covered by outer region 16x16 blocks.

    Returns:
        all_outer_base_patches: Set of (i, j) positions covered by outer patches
    """
    all_outer_base_patches = set()
    for i, j in outer_patches:
        for di in range(2):
            for dj in range(2):
                all_outer_base_patches.add((i + di, j + dj))
    return all_outer_base_patches


def visualize_foveated_vision(save_path='myplot.png'):
    """
    Generate the foveated vision region division visualization.

    Args:
        save_path: Path to save the figure
    """
    grid_size = 28
    center = 13.5
    R_keep = 13.5
    R_center = 10.5

    # Compute regions
    center_patches, outer_patches, discarded_patches = compute_foveated_regions()
    all_outer_base_patches = get_outer_patch_coverage(outer_patches)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors
    colors = {
        'center': '#E74C3C',      # Red for center
        'outer': '#F1C40F',       # Yellow for outer
        'discarded': '#BDC3C7',  # Gray for discarded
    }

    # Draw all patches
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in center_patches:
                color = colors['center']
            elif (i, j) in all_outer_base_patches:
                color = colors['outer']
            else:
                color = colors['discarded']

            rect = mpatches.Rectangle(
                (j, grid_size - 1 - i),  # (x, y), y is inverted for image coords
                1, 1,
                linewidth=0.5,
                edgecolor='white',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)

    # Draw circular boundaries
    circle_outer = plt.Circle(
        (center, center), R_keep,
        fill=False, color='black', linewidth=2.5, linestyle='-'
    )
    ax.add_patch(circle_outer)

    circle_inner = plt.Circle(
        (center, center), R_center,
        fill=False, color='black', linewidth=2.5, linestyle='--'
    )
    ax.add_patch(circle_inner)

    # Add radius annotations
    ax.annotate(
        'r = 10.5',
        xy=(center + R_center * 0.7, center + R_center * 0.7),
        xytext=(center + R_center * 0.7 + 2, center + R_center * 0.7 + 2),
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )

    ax.annotate(
        'r = 13.5',
        xy=(center + R_keep * 0.9, center + R_keep * 0.9),
        xytext=(center + R_keep * 0.9 + 1.5, center + R_keep * 0.9 + 1.5),
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )

    # Set axis properties
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect('equal')

    # Labels and title
    ax.set_xlabel('Patch Column (j)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Patch Row (i)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Foveated Vision: Region Division\n'
        '(224×224 Image → 28×28 Patches)',
        fontsize=16, fontweight='bold', pad=20
    )

    # Set tick marks
    ax.set_xticks(np.arange(0, grid_size + 1, 4))
    ax.set_yticks(np.arange(0, grid_size + 1, 4))
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Create legend
    legend_patches = [
        mpatches.Patch(color=colors['center'], label=f'Center: 8×8 patch (≤ {R_center})', alpha=0.8),
        mpatches.Patch(color=colors['outer'], label=f'Outer: 16×16 merged (10.5 < r ≤ {R_keep})', alpha=0.8),
        mpatches.Patch(color=colors['discarded'], label=f'Discarded: r > {R_keep}', alpha=0.8),
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=11,
        framealpha=0.9,
        title='Region Types',
        title_fontsize=12
    )

    # Add statistics text box
    stats_text = (
        f'Token Statistics:\n'
        f'━━━━━━━━━━━━━━━━\n'
        f'Center patches: {len(center_patches)}\n'
        f'Outer patches: {len(outer_patches)}\n'
        f'Total tokens: {len(center_patches) + len(outer_patches)}\n'
        f'Discarded: {len(discarded_patches)}'
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        1.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=props,
        family='monospace'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualization saved to: {save_path}")
    print(f"\nStatistics:")
    print(f"  Center region (8x8 patches): {len(center_patches)}")
    print(f"  Outer region (16x16 merged): {len(outer_patches)}")
    print(f"  Total tokens: {len(center_patches) + len(outer_patches)}")
    print(f"  Discarded patches: {len(discarded_patches)}")

    return {
        'center_patches': len(center_patches),
        'outer_patches': len(outer_patches),
        'total_tokens': len(center_patches) + len(outer_patches),
        'discarded': len(discarded_patches)
    }


def visualize_with_original_comparison(save_path='comparison_plot.png'):
    """
    Generate a comparison visualization showing:
    1. Standard ViT (28x28 = 784 tokens)
    2. Foveated ViT (~403 tokens)
    """
    grid_size = 28
    center = 13.5
    R_keep = 13.5
    R_center = 10.5

    center_patches, outer_patches, discarded_patches = compute_foveated_regions()
    all_outer_base_patches = get_outer_patch_coverage(outer_patches)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Colors
    colors = {
        'center': '#E74C3C',
        'outer': '#F1C40F',
        'discarded': '#BDC3C7',
        'standard': '#3498DB',
    }

    # Left: Standard ViT
    ax1 = axes[0]
    for i in range(grid_size):
        for j in range(grid_size):
            rect = mpatches.Rectangle(
                (j, grid_size - 1 - i), 1, 1,
                linewidth=0.5, edgecolor='white',
                facecolor=colors['standard'], alpha=0.8
            )
            ax1.add_patch(rect)

    ax1.set_xlim(-0.5, grid_size + 0.5)
    ax1.set_ylim(-0.5, grid_size + 0.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Patch Column (j)', fontsize=12)
    ax1.set_ylabel('Patch Row (i)', fontsize=12)
    ax1.set_title(
        f'Standard ViT\n'
        f'(28×28 = 784 tokens)',
        fontsize=14, fontweight='bold'
    )
    ax1.set_xticks(np.arange(0, grid_size + 1, 4))
    ax1.set_yticks(np.arange(0, grid_size + 1, 4))

    # Add token count
    ax1.text(
        0.5, -0.12,
        'All 784 patches are used',
        transform=ax1.transAxes,
        fontsize=11, ha='center', style='italic'
    )

    # Right: Foveated ViT
    ax2 = axes[1]
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in center_patches:
                color = colors['center']
            elif (i, j) in all_outer_base_patches:
                color = colors['outer']
            else:
                color = colors['discarded']

            rect = mpatches.Rectangle(
                (j, grid_size - 1 - i), 1, 1,
                linewidth=0.5, edgecolor='white',
                facecolor=color, alpha=0.8
            )
            ax2.add_patch(rect)

    # Draw circles
    circle_outer = plt.Circle((center, center), R_keep, fill=False,
                               color='black', linewidth=2, linestyle='-')
    ax2.add_patch(circle_outer)
    circle_inner = plt.Circle((center, center), R_center, fill=False,
                               color='black', linewidth=2, linestyle='--')
    ax2.add_patch(circle_inner)

    ax2.set_xlim(-0.5, grid_size + 0.5)
    ax2.set_ylim(-0.5, grid_size + 0.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Patch Column (j)', fontsize=12)
    ax2.set_ylabel('Patch Row (i)', fontsize=12)
    ax2.set_title(
        f'Foveated ViT\n'
        f'({len(center_patches) + len(outer_patches)} tokens)',
        fontsize=14, fontweight='bold'
    )
    ax2.set_xticks(np.arange(0, grid_size + 1, 4))
    ax2.set_yticks(np.arange(0, grid_size + 1, 4))

    # Add reduction percentage
    reduction = (1 - (len(center_patches) + len(outer_patches)) / 784) * 100
    ax2.text(
        0.5, -0.12,
        f'~{reduction:.1f}% token reduction',
        transform=ax2.transAxes,
        fontsize=11, ha='center', style='italic', color='red'
    )

    # Create legend
    legend_patches = [
        mpatches.Patch(color=colors['center'], label='Center (8×8)', alpha=0.8),
        mpatches.Patch(color=colors['outer'], label='Outer (16×16)', alpha=0.8),
        mpatches.Patch(color=colors['discarded'], label='Discarded', alpha=0.8),
    ]
    ax2.legend(handles=legend_patches, loc='upper left', fontsize=10)

    plt.suptitle(
        'Foveated Vision: Token Reduction Comparison',
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison visualization saved to: {save_path}")


if __name__ == '__main__':
    print("Generating Foveated Vision visualizations...\n")
    print("=" * 50)

    # Generate main visualization
    stats = visualize_foveated_vision('myplot.png')

    # Generate comparison visualization
    visualize_with_original_comparison('comparison_plot.png')

    print("\n" + "=" * 50)
    print("All visualizations generated successfully!")

# coding=utf-8
"""
Foveated Vision Transformer (FoveatedViT)

This module implements a foveated vision mechanism for Vision Transformer,
simulating human eye's "central clear, peripheral blur" visual characteristic.

Core idea:
- Circular cropping: Keep only patches within radius R_keep = 13.5
- Center region (r <= 10.5): 8x8 patches, high resolution
- Outer region (10.5 < r <= 13.5): 16x16 merged patches, lower resolution

This reduces token count while maintaining high resolution at the center.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm
from torch.nn.modules.utils import _pair

import models.configs as configs
from .modeling_resnet import ResNetV2
import ml_collections


logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class FoveatedPatchEmbed(nn.Module):
    """
    Foveated Patch Embedding Layer

    Implements the foveated vision mechanism:
    1. Circular cropping based on distance from center
    2. Center region: 8x8 patches (high resolution)
    3. Outer region: 2x2 patches merged to 16x16 (lower resolution)
    """

    def __init__(self, config, img_size=224, base_patch_size=8):
        super(FoveatedPatchEmbed, self).__init__()
        self.img_size = img_size
        self.base_patch_size = base_patch_size
        self.hidden_size = config.hidden_size

        # Grid dimensions: 224 / 8 = 28
        self.grid_size = img_size // base_patch_size  # 28
        self.num_base_patches = self.grid_size * self.grid_size  # 784

        # Foveated vision parameters
        self.center = 13.5  # Center coordinates (in patch units)
        self.R_keep = 13.5  # Keep radius (in patch units)
        self.R_center = 10.5  # Center region radius (r <= R_center: 8x8 patches)
        # Outer region: R_center < r <= R_keep: 16x16 merged patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Dropout
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # Projection layers for foveated embedding (defined once, reused every forward)
        # Center: 8x8 patches -> 192 -> hidden_size
        center_in_features = base_patch_size * base_patch_size * 3  # 8*8*3 = 192
        self.center_proj = nn.Linear(center_in_features, config.hidden_size)

        # Outer: 16x16 patches = 4x 8x8 -> 768 -> hidden_size
        outer_in_features = base_patch_size * base_patch_size * 3 * 4  # 8*8*3*4 = 768
        self.outer_proj = nn.Linear(outer_in_features, config.hidden_size)

        # Precompute patch classification
        self._compute_patch_info()

    def _compute_patch_info(self):
        """
        Precompute which patches belong to center, outer, or discarded regions.
        Also compute the merged patch mapping for outer region.
        """
        self.center_patches = []  # List of (i, j) positions for 8x8 patches
        self.outer_patches = []  # List of (i, j) positions for 16x16 patches
        self.discarded_patches = []  # List of (i, j) positions to discard

        # Mapping from base patch position to region type
        self.patch_region = {}  # (i, j) -> 'center', 'outer', or 'discarded'
        self.patch_to_outer_token = {}  # (i, j) -> outer token index

        outer_token_idx = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Distance from center (using center of patch coordinates)
                dist = math.sqrt((i - self.center) ** 2 + (j - self.center) ** 2)

                if dist <= self.R_keep:
                    # Patch is within the circular region
                    if dist <= self.R_center:
                        # Center region: 8x8 patches
                        self.center_patches.append((i, j))
                        self.patch_region[(i, j)] = 'center'
                    else:
                        # Outer region: 16x16 merged patches
                        # Only keep top-left patch of each 2x2 block
                        if i % 2 == 0 and j % 2 == 0:
                            # Check if all 4 patches in the 2x2 block are valid
                            block_valid = True
                            for di in range(2):
                                for dj in range(2):
                                    ni, nj = i + di, j + dj
                                    if ni >= self.grid_size or nj >= self.grid_size:
                                        block_valid = False
                                        break
                                    block_dist = math.sqrt((ni - self.center) ** 2 + (nj - self.center) ** 2)
                                    if block_dist > self.R_keep:
                                        block_valid = False
                                        break
                                if not block_valid:
                                    break

                            if block_valid:
                                self.outer_patches.append((i, j))
                                self.patch_to_outer_token[(i, j)] = outer_token_idx
                                outer_token_idx += 1
                                for di in range(2):
                                    for dj in range(2):
                                        ni, nj = i + di, j + dj
                                        self.patch_region[(ni, nj)] = 'outer'
                else:
                    # Discarded patch
                    self.discarded_patches.append((i, j))
                    self.patch_region[(i, j)] = 'discarded'

        # Token counts
        self.num_center_tokens = len(self.center_patches)
        self.num_outer_tokens = len(self.outer_patches)
        self.num_total_tokens = self.num_center_tokens + self.num_outer_tokens

        # Create mapping from base patch position to final token index
        self.patch_to_token_idx = {}
        for idx, (i, j) in enumerate(self.center_patches):
            self.patch_to_token_idx[(i, j)] = idx
        for idx, (i, j) in enumerate(self.outer_patches):
            # Outer tokens come after center tokens
            self.patch_to_token_idx[(i, j)] = self.num_center_tokens + idx

        # Create indices for gathering center region embeddings
        self.center_patch_indices = torch.tensor(
            [i * self.grid_size + j for i, j in self.center_patches],
            dtype=torch.long
        )

        # Create indices for gathering outer region patches (each outer patch has 4 base patches)
        outer_patch_indices = []
        for i, j in self.outer_patches:
            for di in range(2):
                for dj in range(2):
                    outer_patch_indices.append(i * self.grid_size + j + di * self.grid_size + dj)
        self.outer_patch_indices = torch.tensor(outer_patch_indices, dtype=torch.long)

        # Register buffers for efficient indexing
        self.register_buffer('center_idx', self.center_patch_indices)
        self.register_buffer('outer_idx', self.outer_patch_indices)

    def forward(self, x):
        """
        Forward pass for foveated patch embedding.

        Args:
            x: Input tensor of shape (B, C, H, W) = (B, 3, 224, 224)

        Returns:
            embeddings: Tensor of shape (B, num_tokens, hidden_size)
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Expected {self.img_size}x{self.img_size} input"

        # Reshape to base patches: (B, 3, 28, 8, 28, 8) -> (B, 28, 8, 28, 8, 3)
        # Then transpose to (B, 28, 28, 8, 8, 3)
        x_reshaped = x.view(B, C, self.grid_size, self.base_patch_size,
                            self.grid_size, self.base_patch_size)
        x_reshaped = x_reshaped.permute(0, 2, 4, 3, 5, 1)  # (B, 28, 28, 8, 8, 3)
        # Flatten patches: (B, 28, 28, 8*8*3) = (B, 28, 28, 192)
        x_patches = x_reshaped.reshape(B, self.grid_size * self.grid_size, -1)

        # Create embeddings for center region (8x8 patches)
        if self.num_center_tokens > 0:
            center_patches = x_patches[:, self.center_idx, :]  # (B, num_center, 192)
            center_embeddings = self.center_proj(center_patches)  # (B, num_center, hidden)
        else:
            center_embeddings = torch.empty(B, 0, self.hidden_size, device=x.device, dtype=x.dtype)

        # Create embeddings for outer region (16x16 patches = 4x 8x8 patches concatenated)
        if self.num_outer_tokens > 0:
            # Gather outer patches: (B, num_outer * 4, 192)
            outer_patches = x_patches[:, self.outer_idx, :]
            # Reshape to (B, num_outer, 4, 192) and concatenate
            outer_patches = outer_patches.view(B, self.num_outer_tokens, 4, -1)
            outer_patches = outer_patches.reshape(B, self.num_outer_tokens, -1)  # (B, num_outer, 768)
            # Project to hidden size (16*16*3 = 768 -> hidden_size)
            outer_embeddings = self.outer_proj(outer_patches)  # (B, num_outer, hidden)
        else:
            outer_embeddings = torch.empty(B, 0, self.hidden_size, device=x.device, dtype=x.dtype)

        # Concatenate center and outer embeddings
        x = torch.cat([center_embeddings, outer_embeddings], dim=1)  # (B, num_total, hidden)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_total + 1, hidden)

        x = self.dropout(x)
        return x

    def get_num_tokens(self):
        """Return the number of tokens (excluding cls token)."""
        return self.num_total_tokens


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FoveatedEmbeddings(nn.Module):
    """Construct the embeddings from foveated patches, position embeddings."""

    def __init__(self, config, img_size=224, base_patch_size=8):
        super(FoveatedEmbeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        # Initialize foveated patch embedding
        self.patch_embeddings = FoveatedPatchEmbed(
            config, img_size=img_size[0], base_patch_size=base_patch_size
        )
        self.num_patches = self.patch_embeddings.get_num_tokens()

        # Create position embeddings
        # We'll interpolate from the standard 28x28 position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size)
        )
        self.cls_token = self.patch_embeddings.cls_token

        self.dropout = Dropout(config.transformer["dropout_rate"])

        # Initialize position embeddings with interpolation
        self._init_position_embeddings(config)

    def _init_position_embeddings(self, config):
        """
        Initialize position embeddings by interpolating from standard 28x28 grid.
        """
        # Standard 28x28 position embeddings
        grid_size = 28  # 224 / 8 = 28
        num_patches_standard = grid_size * grid_size

        # Create standard position embeddings
        pos_embed = self._create_sinusoidal_positions(
            num_positions=num_patches_standard + 1,
            embed_dim=config.hidden_size
        )

        # Interpolate to foveated positions
        foveated_pos_embed = self._interpolate_position_embeddings(
            pos_embed, grid_size, config.hidden_size
        )

        with torch.no_grad():
            self.position_embeddings.copy_(foveated_pos_embed)

    def _create_sinusoidal_positions(self, num_positions, embed_dim):
        """Create sinusoidal position embeddings."""
        pe = torch.zeros(num_positions, embed_dim)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _interpolate_position_embeddings(self, pos_embed, grid_size, embed_dim):
        """
        Interpolate position embeddings for the foveated grid.

        For center patches, use their original positions.
        For outer patches, use the average of their 4 component positions.
        """
        B, N, C = 1, self.num_patches + 1, embed_dim
        foveated_pos_embed = torch.zeros(B, N, C)

        # CLS token position
        foveated_pos_embed[0, 0] = pos_embed[0, 0]

        patch_emb = self.patch_embeddings

        # Center patches: direct mapping
        for idx, (i, j) in enumerate(patch_emb.center_patches):
            # Convert 2D position to 1D index
            orig_idx = i * grid_size + j + 1  # +1 for CLS token
            foveated_pos_embed[0, 1 + idx] = pos_embed[0, orig_idx]

        # Outer patches: average of 2x2 block positions
        for idx, (i, j) in enumerate(patch_emb.outer_patches):
            # Get average position of the 2x2 block
            total_pos = torch.zeros(C)
            count = 0
            for di in range(2):
                for dj in range(2):
                    ni, nj = i + di, j + dj
                    orig_idx = ni * grid_size + nj + 1
                    total_pos += pos_embed[0, orig_idx]
                    count += 1
            foveated_pos_embed[0, 1 + patch_emb.num_center_tokens + idx] = total_pos / count

        return foveated_pos_embed

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)

        # Add position embeddings (already added in patch_embeddings for cls_token)
        # For remaining tokens:
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, base_patch_size=8):
        super(Transformer, self).__init__()
        self.embeddings = FoveatedEmbeddings(config, img_size=img_size, base_patch_size=base_patch_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class FoveatedViT(nn.Module):
    """
    Foveated Vision Transformer

    A variant of ViT that simulates foveated vision:
    - Center region: high resolution (8x8 patches)
    - Outer region: lower resolution (16x16 merged patches)
    - Circular cropping removes corner patches
    """
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 base_patch_size=8):
        super(FoveatedViT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis, base_patch_size=base_patch_size)
        self.head = Linear(config.hidden_size, num_classes)

        # Store info for visualization/debugging
        self.num_tokens = self.transformer.embeddings.patch_embeddings.num_total_tokens
        self.num_center_tokens = self.transformer.embeddings.patch_embeddings.num_center_tokens
        self.num_outer_tokens = self.transformer.embeddings.patch_embeddings.num_outer_tokens

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        """Load pretrained weights (for standard ViT)."""
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            # Note: patch_embeddings are different, so we don't load them directly
            # Instead, they are randomly initialized
            self.transformer.embeddings.patch_embeddings.cls_token.copy_(
                np2th(weights["cls"])
            )
            self.transformer.embeddings.position_embeddings.copy_(
                np2th(weights["Transformer/posembed_input/pos_embedding"])
            )
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            # Load transformer blocks
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

    def get_token_info(self):
        """Return token information for debugging/visualization."""
        return {
            'num_center_tokens': self.num_center_tokens,
            'num_outer_tokens': self.num_outer_tokens,
            'num_total_tokens': self.num_tokens,
            'center_patches': self.transformer.embeddings.patch_embeddings.center_patches,
            'outer_patches': self.transformer.embeddings.patch_embeddings.outer_patches,
            'discarded_patches': self.transformer.embeddings.patch_embeddings.discarded_patches,
            'grid_size': self.transformer.embeddings.patch_embeddings.grid_size,
        }


# Configuration factory for FoveatedViT
def get_foveated_b16_config():
    """Returns the FoveatedViT-B/16 configuration (using 8x8 base patches)."""
    config = configs.get_b16_config()
    # Patch size is 8x8 for foveated version
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    return config


def get_foveated_b16_full_config():
    """Returns a FoveatedViT configuration with same capacity as ViT-B/16.

    Maintains the foveated sampling design:
    - Center region: 8x8 patches (high resolution)
    - Outer region: 16x16 merged patches (lower resolution)
    - Circular cropping with R_keep=13.5

    Architecture matches ViT-B/16:
    - hidden_size: 768
    - num_layers: 12
    - num_heads: 12
    - mlp_dim: 3072 (mlp_ratio=4.0)
    """
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072  # hidden_size * 4
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    # Foveated patch size: 8x8 base patches
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    return config


def get_foveated_b16_small_config():
    """Returns a smaller FoveatedViT configuration for faster training."""
    config = configs.get_testing()
    config.hidden_size = 384
    config.transformer.num_heads = 6
    config.transformer.num_layers = 6
    config.transformer.mlp_dim = 1536
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    return config


CONFIGS = {
    'FoveatedViT-B_8': get_foveated_b16_config(),
    'FoveatedViT-B_8_full': get_foveated_b16_full_config(),
    'FoveatedViT-Small_8': get_foveated_b16_small_config(),
}

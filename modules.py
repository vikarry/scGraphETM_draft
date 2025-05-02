import datetime
import os
import shutil
import warnings
import gc
from typing import Tuple, List, Optional, Dict, Any
import argparse
import datetime
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import SAGEConv
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import scanpy as sc
from tqdm import tqdm
import pybedtools
import pickle

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scDataLoader import ScDataLoader, set_seed
from preprocess.preprocess import run_preprocessing

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SelfAttention(nn.Module):
    """Self-attention mechanism."""

    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.query = nn.Linear(input_dim, emb_size)
        self.key = nn.Linear(input_dim, emb_size)
        self.value = nn.Linear(input_dim, emb_size)
        self.linear_o1 = nn.Linear(emb_size, emb_size)
        self.scale = emb_size ** -0.5
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        self.attention_weights = F.softmax(attention_scores, dim=-1)

        return torch.matmul(self.attention_weights, V)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout."""

    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_size),
            nn.LayerNorm(emb_size),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + residual


class VAE(nn.Module):
    """Variational Autoencoder with self-attention and residual blocks."""

    def __init__(self, input_dim: int, emb_size: int, num_topics: int, block_num: int = 1):
        super().__init__()
        self.attention1 = SelfAttention(input_dim=input_dim, emb_size=emb_size)
        self.residual_block1 = ResidualBlock(input_dim, emb_size)
        self.blocks = nn.ModuleList([
            ResidualBlock(emb_size, emb_size) for _ in range(block_num)
        ])
        self.mu = nn.Linear(emb_size, num_topics, bias=False)
        self.log_sigma = nn.Linear(emb_size, num_topics, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.attention1(x)
        h = self.residual_block1(x, residual)

        for block in self.blocks:
            h = block(h, residual)
            residual = h

        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        kl_theta = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1).mean()

        return mu, log_sigma, kl_theta


class LDEC(nn.Module):
    """Linear Decoder with optional GNN transformation and parameter sharing."""

    def __init__(self, num_modality: int, emb_size: int, num_topics: int, batch_size: int,
                 shared_module: Optional[nn.Module] = None):
        super().__init__()

        # Create or use shared alpha transformation
        if shared_module is not None:
            self.register_module('alphas', shared_module)
        else:
            alphas = nn.Sequential(
                nn.Linear(emb_size, num_topics, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
            )
            self.register_module('alphas', alphas)

        # These parameters remain independent for each decoder
        self.rho = nn.Parameter(torch.randn(num_modality, emb_size))
        self.batch_bias = nn.Parameter(torch.randn(batch_size, num_modality))
        self.beta = None

    def forward(self, theta: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        matrix = self.rho if matrix is None else matrix
        beta = F.softmax(self.alphas(matrix), dim=0).transpose(1, 0)
        self.beta = beta
        res = torch.mm(theta, beta)
        return torch.log(res + 1e-6)


class GNN(nn.Module):
    """Graph Neural Network using GraphSAGE convolutions."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return F.leaky_relu(x)


class GeneEmbedding(nn.Module):
    """Gene embedding module using xTrimoGene's auto-discretization approach."""

    def __init__(self,
                 num_genes: int,  # Number of genes
                 emb_dim: int,  # Embedding dimension
                 num_bins: int = 100,  # Number of bins for auto-discretization
                 node2vec: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_genes = num_genes
        self.emb_dim = emb_dim
        self.num_bins = num_bins

        self.exp_lookup = nn.Parameter(torch.randn(num_bins, emb_dim) / np.sqrt(emb_dim))

        # First projection layer
        self.projection = nn.Sequential(
            nn.Linear(1, emb_dim),  # Project scalar values to embedding dimension
            nn.LayerNorm(emb_dim),  # Add layer normalization for stability
            nn.LeakyReLU()  # Non-linearity as per paper
        )

        # Cross-layer projection weights
        self.w2 = nn.Parameter(torch.randn(emb_dim, num_bins) / np.sqrt(emb_dim))  # Changed dimension
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Final transformation layer
        self.output_transform = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.LeakyReLU()
        )

        if node2vec is not None:
            self.register_buffer('G', node2vec)
        else:
            self.register_buffer('G', torch.zeros(num_genes, emb_dim))

        # Initialize Linear layers properly
        self._init_parameters()

    def _init_parameters(self):
        """Initialize only Linear layer parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def auto_discretize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_genes) or (num_genes,)

        Returns:
            Discretized embeddings of shape (batch_size, num_genes, emb_dim)
        """
        # Ensure x is a proper tensor and has the right shape
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a tensor, got {type(x)}")

        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        # Ensure shape is correct
        if x.size(-1) != self.num_genes:
            x = x.transpose(-1, -2)

        batch_size = x.size(0)

        # Add feature dimension for projection
        x = x.unsqueeze(-1)  # Shape: (batch_size, num_genes, 1)

        # First projection and non-linearity
        v1 = self.projection(x)  # Shape: (batch_size, num_genes, emb_dim)

        # Cross-layer projection with mixing factor
        # v1 shape: (batch_size, num_genes, emb_dim)
        # w2 shape: (emb_dim, num_bins)
        v2 = torch.matmul(v1, self.w2)  # Shape: (batch_size, num_genes, num_bins)
        v2 = v2 + self.alpha * v1.matmul(torch.ones_like(self.w2))  # Same shape as v2

        # Generate bin weights using softmax
        bin_weights = F.softmax(v2, dim=-1)  # Shape: (batch_size, num_genes, num_bins)

        # Get embeddings from lookup table
        # bin_weights shape: (batch_size, num_genes, num_bins)
        # exp_lookup shape: (num_bins, emb_dim)
        E = torch.matmul(bin_weights, self.exp_lookup)  # Shape: (batch_size, num_genes, emb_dim)

        return E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate gene embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_genes) or (num_genes,)

        Returns:
            Gene embeddings of shape (batch_size, num_genes, emb_dim)
        """
        E = self.auto_discretize(x)
        E = self.output_transform(E)

        output = E + self.G[None, :, :]

        return output

class AtacEmbedding(nn.Module):
    """Learn embeddings for ATAC peaks using attention mechanism."""

    def __init__(self, b: int, d: int, node2vec: Optional[torch.Tensor] = None):
        """
        Args:
            b: Input dimension (number of peaks)
            d: Output embedding dimension
            node2vec: Pre-trained node2vec embeddings (optional)
        """
        super(AtacEmbedding, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1, b))  # 1xb
        self.w2 = nn.Parameter(torch.randn(b, b))  # bxb
        self.alpha = nn.Parameter(torch.randn(b))  # b
        self.T = nn.Parameter(torch.randn(b, d))  # bxd
        self.register_buffer('G', node2vec if node2vec is not None else torch.zeros(b, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ATAC embedding.
        Args:
            x: Input tensor of shape (batch_size, num_peaks)
        Returns:
            ATAC embeddings of shape (batch_size, embedding_dim)
        """
        # Attention mechanism
        a = F.leaky_relu(self.w1.expand(x.size(1), self.w1.size(1)))  # bx1
        z = a @ self.w2 + self.alpha * a  # bxb

        gamma = F.softmax(z, dim=1)  # bxb

        E = torch.matmul(gamma, self.T)  # bxd
        atac_embeddings = E + self.G  # Add pre-trained embeddings if available

        return atac_embeddings


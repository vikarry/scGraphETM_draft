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

from modules import SelfAttention, ResidualBlock, VAE, LDEC, GNN, GeneEmbedding, AtacEmbedding


class ScModel(nn.Module):
    def __init__(self, num_of_gene: int, num_of_peak: int, num_of_gene_gnn: int,
                 num_of_peak_gnn: int, emb_size: int, num_of_topic: int, device: torch.device,
                 batch_size: int, result_save_path: Tuple[str, str], gnn_conv: Optional[str] = None,
                 lr: float = 0.001, best_model_path: Optional[str] = None,
                 use_graph_recon: bool = False, graph_recon_weight: float = 1.0,
                 pos_weight: float = 1.0, node2vec: Optional[torch.Tensor] = None,
                 latent_size: int = 100, use_gnn: bool = True, use_xtrimo: bool = False, shared_decoder: bool = False):
        super().__init__()
        self.device = device
        self.use_graph_recon = use_graph_recon
        self.graph_recon_weight = graph_recon_weight
        self.pos_weight = pos_weight
        self.use_gnn = use_gnn
        self.shared_decoder = shared_decoder
        self.emb_size = emb_size

        # Initialize VAEs
        self.vae_rna = VAE(num_of_gene, emb_size, num_of_topic).to(device)
        self.vae_atac = VAE(num_of_peak, emb_size, num_of_topic).to(device)

        # Initialize GNN if needed
        self.gnn = GNN(emb_size, emb_size * 2, emb_size).to(device) if use_gnn else None

        # Initialize decoders with shared parameters if requested
        if shared_decoder:
            # Create shared alpha transformation
            shared_alphas = nn.Sequential(
                nn.Linear(emb_size, num_of_topic, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
            ).to(device)

            # Create decoders with shared alphas
            self.decoder_rna = LDEC(num_of_gene, emb_size, num_of_topic, batch_size,
                                    shared_module=shared_alphas).to(device)
            self.decoder_atac = LDEC(num_of_peak, emb_size, num_of_topic, batch_size,
                                     shared_module=shared_alphas).to(device)
        else:
            # Create independent decoders
            self.decoder_rna = LDEC(num_of_gene, emb_size, num_of_topic, batch_size).to(device)
            self.decoder_atac = LDEC(num_of_peak, emb_size, num_of_topic, batch_size).to(device)

        # Store all models in ModuleList for consistency
        self.models = nn.ModuleList([
            self.vae_rna,
            self.vae_atac,
            self.gnn,
            self.decoder_rna,
            self.decoder_atac,
        ])
        self.use_xtrimo = use_xtrimo
        self.node2vec = node2vec
        self.gene_node2vec, self.peak_node2vec = (
            self.split_tensor(node2vec, num_of_gene) if node2vec is not None
            else (
                torch.randn(num_of_gene, emb_size),  # Random matrix for genes
                torch.randn(num_of_peak, emb_size)  # Random matrix for peaks
            )
        )
        if use_xtrimo:

            self.gene_embedding = GeneEmbedding(
                num_genes=num_of_gene,
                emb_dim=emb_size,
                num_bins=100,  # Can be adjusted
                node2vec=self.gene_node2vec
            ).to(device)

            self.atac_embedding = GeneEmbedding(
                num_genes=num_of_peak,
                emb_dim=emb_size,
                num_bins=100,  # Can be adjusted
                node2vec=self.peak_node2vec
            ).to(device)

        else:
            self.gene_embedding = None
            self.atac_embedding = None

        # Rest of initialization...
        self.setup_network_components(num_of_gene, num_of_peak, num_of_gene_gnn,
                                      num_of_peak_gnn, emb_size, batch_size)
        self.setup_training_components(lr)
        self.initialize_metrics()

        self.umap_dir, self.tsne_dir = result_save_path

        if best_model_path:
            self.load_best_model(best_model_path)

    def setup_network_components(self, num_of_gene: int, num_of_peak: int,
                                 num_of_gene_gnn: int, num_of_peak_gnn: int,
                                 emb_size: int, batch_size: int):
        """Setup neural network components."""
        self.batch_to_emb = nn.Sequential(
            nn.Linear(batch_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.LeakyReLU(),
        ).to(self.device)

        self.lin_rna = nn.Sequential(
            nn.Linear(num_of_gene_gnn, num_of_gene),
            nn.BatchNorm1d(num_of_gene),
            nn.LeakyReLU(),
        ).to(self.device)

        self.lin_peak = nn.Sequential(
            nn.Linear(num_of_peak_gnn, num_of_peak),
            nn.BatchNorm1d(num_of_peak),
            nn.LeakyReLU(),
        ).to(self.device)

    def setup_training_components(self, lr: float):
        """Setup training-related components."""
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1.2e-6)
        self.scaler = StandardScaler()
        self.loss_scaler = GradScaler()

    def initialize_metrics(self):
        """Initialize model metrics."""
        self.best_emb = None
        self.best_model = None
        self.best_train_ari = 0
        self.best_test_ari = 0
        self.epoch = 0

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def compute_graph_reconstruction_loss(self, emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute graph reconstruction loss using embeddings and edge index.
        Args:
            emb: Node embeddings tensor from GNN of shape (num_nodes, emb_size)
            edge_index: Edge index tensor of shape (2, num_edges)
        Returns:
            Reconstruction loss computed on edges
        """
        num_pos_edges = edge_index.shape[1]

        # Get source and target node embeddings for positive edges
        src_emb = emb[edge_index[0]]  # Shape: (num_pos_edges, emb_size)
        dst_emb = emb[edge_index[1]]  # Shape: (num_pos_edges, emb_size)
        pos_logits = torch.sum(src_emb * dst_emb, dim=1)  # Shape: (num_pos_edges,)

        # Sample random node pairs for negative edges (same number as positive edges)
        num_nodes = emb.shape[0]
        neg_src = torch.randint(0, num_nodes, (num_pos_edges,), device=self.device)
        neg_dst = torch.randint(0, num_nodes, (num_pos_edges,), device=self.device)

        # Get embeddings for negative edges
        neg_src_emb = emb[neg_src]  # Shape: (num_pos_edges, emb_size)
        neg_dst_emb = emb[neg_dst]  # Shape: (num_pos_edges, emb_size)
        neg_logits = torch.sum(neg_src_emb * neg_dst_emb, dim=1)  # Shape: (num_pos_edges,)

        # Combine positive and negative logits
        logits = torch.cat([pos_logits, neg_logits])  # Shape: (2*num_pos_edges,)

        # Create labels (1 for positive edges, 0 for negative edges)
        labels = torch.zeros(2 * num_pos_edges, dtype=torch.float32, device=self.device)
        labels[:num_pos_edges] = 1.0

        # Compute binary cross entropy loss with edge weighting
        pos_weight = torch.tensor([self.pos_weight], device=self.device)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction='mean'
        )

        return loss

    def forward(self, batch: List[torch.Tensor], train_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, ...]:
        batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor)
                 else tensor for tensor in batch]

        (RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
         RNA_tensor_gnn, RNA_tensor_normalized_gnn,
         ATAC_tensor_gnn, ATAC_tensor_normalized_gnn) = batch

        encoder1, encoder2, gnn, decoder1, decoder2 = self.models

        # Process RNA data
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        z1 = self.reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        # Process ATAC data
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)
        z2 = self.reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        # Initialize variables
        rho, eta = None, None
        edge_recon_loss = torch.tensor(0.0, device=self.device)
        emb = None

        # GNN processing if enabled
        if self.use_gnn:
            edge_index = self.get_edge_index(train_loader)

            if self.use_xtrimo:
                # Use learned embeddings
                gene_embeddings = self.gene_embedding(RNA_tensor_normalized)
                atac_embeddings = self.atac_embedding(ATAC_tensor_normalized)
                fm = torch.cat((
                    gene_embeddings.mean(dim=0),
                    atac_embeddings.mean(dim=0)
                ), dim=0)
            else:
                specific_fm = torch.cat((ATAC_tensor_normalized.T, RNA_tensor_normalized.T), dim=0)
                specific_fm = self.batch_to_emb(specific_fm).to(self.device)
                fm = specific_fm * self.node2vec.to(self.device)

            # Pass through GNN
            emb = gnn(fm, edge_index)
            rho, eta = self.split_tensor(emb, RNA_tensor_gnn.shape[1])

            # Compute graph reconstruction loss if enabled
            if self.use_graph_recon:
                edge_recon_loss = self.compute_graph_reconstruction_loss(emb, edge_index)
                edge_recon_loss = edge_recon_loss * self.graph_recon_weight

        # Decode - pass None for matrix if GNN is disabled
        pred_RNA_tensor = decoder1(theta1, rho if self.use_gnn else None)
        pred_ATAC_tensor = decoder2(theta2, eta if self.use_gnn else None)

        # Calculate losses
        recon_loss1 = -(pred_RNA_tensor * RNA_tensor).sum(-1)
        recon_loss2 = -(pred_ATAC_tensor * ATAC_tensor).sum(-1)
        recon_loss = (recon_loss1 + recon_loss2).mean()
        kl_loss = (kl_theta1 + kl_theta2).mean()

        return recon_loss, kl_loss, edge_recon_loss, emb

    def train_epoch(self, data_loader: ScDataLoader, epochs: int = 20, finetune: bool = False):
        """Train the model for specified number of epochs."""
        train_loader = data_loader.train_loader
        test_loader = None if finetune else data_loader.test_loader

        for epoch in range(epochs):
            self.epoch = epoch + 1
            print(f"Epoch {epoch + 1}/{epochs}")

            # Calculate weights for different loss components
            recon_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0.6, 8, True)
            kl_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 1e-2, False)
            edge_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 10, False)

            losses = self.train_single_epoch(train_loader, recon_loss_weight,
                                             kl_weight, edge_loss_weight)

            # Print training information
            print(f"recon_loss_weight: {recon_loss_weight}, "
                  f"kl_weight: {kl_weight}, "
                  f"edge_loss_weight: {edge_loss_weight}")
            print(f"Avg Recon Loss: {losses['recon_loss']:.4f}, "
                  f"Avg KL Loss: {losses['kl_loss']:.4f}, "
                  f"Avg edge_recon Loss: {losses['edge_recon_loss']:.4f}, "
                  f"Avg Total Loss: {losses['total_loss']:.4f}")

            if not finetune:
                self.evaluate_and_save(data_loader)

            gc.collect()

        print(f"Best Train ARI: {self.best_train_ari:.4f}, "
              f"Best Test ARI: {self.best_test_ari:.4f}")

    def train_single_epoch(self, train_loader: torch.utils.data.DataLoader,
                           recon_loss_weight: float, kl_weight: float,
                           edge_loss_weight: float) -> Dict[str, float]:
        """Train for a single epoch and return losses."""
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_edge_recon_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            with autocast():
                self.optimizer.zero_grad()
                recon_loss, kl_loss, edge_recon_loss, emb = self.forward(batch, train_loader)
                loss = (recon_loss_weight * recon_loss +
                        kl_loss * kl_weight +
                        edge_loss_weight * edge_recon_loss)

                self.loss_scaler.scale(loss).backward()
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_edge_recon_loss += edge_recon_loss.item()
            total_loss += loss.item()

        return {
            'recon_loss': total_recon_loss / len(train_loader),
            'kl_loss': total_kl_loss / len(train_loader),
            'edge_recon_loss': total_edge_recon_loss / len(train_loader),
            'total_loss': total_loss / len(train_loader)
        }

    def evaluate_and_save(self, data_loader: ScDataLoader):
        """Evaluate model performance and save best model."""
        with autocast():
            train_ari = self.evaluate(
                data_loader.get_all_train_data()['X_rna_tensor_normalized'],
                data_loader.get_all_train_data()['X_atac_tensor_normalized'],
                data_loader.get_all_train_data()['rna_data'],
                True
            )

            test_ari = self.evaluate(
                data_loader.get_all_test_data()['X_rna_tensor_normalized'],
                data_loader.get_all_test_data()['X_atac_tensor_normalized'],
                data_loader.get_all_test_data()['rna_data'],
                False
            )

        if test_ari >= self.best_test_ari:
            self.best_test_ari = test_ari
            self.best_model = self.state_dict()

        if train_ari >= self.best_train_ari:
            self.best_train_ari = train_ari

    @staticmethod
    def calc_weight(epoch: int, n_epochs: int, cutoff_ratio: float,
                    warmup_ratio: float, min_weight: float, max_weight: float,
                    reverse: bool = False) -> float:
        """Calculate weight for loss components based on training progress."""
        if epoch < n_epochs * cutoff_ratio:
            return 0.0

        fully_warmup_epoch = n_epochs * warmup_ratio
        if warmup_ratio:
            if reverse:
                if epoch < fully_warmup_epoch:
                    return 1.0
                weight_progress = min(1.0, (epoch - fully_warmup_epoch) /
                                      (n_epochs - fully_warmup_epoch))
                weight = max_weight - weight_progress * (max_weight - min_weight)
            else:
                weight_progress = min(1.0, epoch / fully_warmup_epoch)
                weight = min_weight + weight_progress * (max_weight - min_weight)
            return max(min_weight, min(max_weight, weight))
        return max_weight

    def get_edge_index(self, train_loader: torch.utils.data.DataLoader,
                       edge_num: Optional[int] = None) -> torch.Tensor:
        """Get edge indices for GNN."""
        edge_index = train_loader.dataset.dataset.edge_index.to(self.device)
        if edge_num:
            selected_edge_index = torch.randperm(edge_index.size(1))[:edge_num]
            edge_index = edge_index[:, selected_edge_index]
        return edge_index

    def get_fm(self, train_loader: torch.utils.data.DataLoader,
               atac_tensor: torch.Tensor, gene_embeddings: torch.Tensor,
               use_rand: bool = False) -> torch.Tensor:
        """Get feature matrix for GNN using gene embeddings."""
        if use_rand:
            return train_loader.dataset.dataset.fm.to(self.device)

        # Use gene embeddings for RNA part of the feature matrix
        fm = torch.cat((gene_embeddings, atac_tensor.T), dim=0)

        return fm

    @staticmethod
    def split_tensor(tensor: torch.Tensor, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into two parts at specified row."""
        if num_rows >= tensor.shape[0]:
            raise ValueError("num_rows should be less than tensor's number of rows")
        return tensor[:num_rows, :], tensor[num_rows:, :]

    def evaluate(self, x1: torch.Tensor, x2: torch.Tensor,
                 adata: ad.AnnData, is_train: bool = True) -> float:
        """Evaluate model performance using clustering metrics."""
        theta = self.get_theta(x1.to(self.device), x2.to(self.device))
        adata.obsm['cell_embed'] = theta.cpu().numpy()
        best_params = self.evaluate_clustering(adata)

        res = f"{best_params['n_neighbors']}_{best_params['method']}_{best_params['resolution']}"
        ari = round(best_params['ari'], 4)
        nmi = round(best_params['nmi'], 4)

        self.save_plots(adata, best_params, 'Train' if is_train else 'Test')
        print(f"{'Train' if is_train else 'Test'} Clustering Info: {res}, ARI: {ari}, NMI: {nmi}")

        return ari

    def get_theta(self, RNA_tensor_normalized: torch.Tensor,
                  ATAC_tensor_normalized: torch.Tensor) -> torch.Tensor:
        """Get theta values from RNA and ATAC data."""
        encoder1, encoder2, _, _, _ = self.models

        with torch.no_grad():
            mu1, log_sigma1, _ = encoder1(RNA_tensor_normalized)
            mu2, log_sigma2, _ = encoder2(ATAC_tensor_normalized)

            z1 = self.reparameterize(mu1, log_sigma1)
            theta1 = F.softmax(z1, dim=-1)

            z2 = self.reparameterize(mu2, log_sigma2)
            theta2 = F.softmax(z2, dim=-1)

            return (theta1 + theta2) / 2

    @staticmethod
    def evaluate_clustering(adata: ad.AnnData) -> Dict[str, Any]:
        """Evaluate clustering performance using different methods and parameters."""
        clustering_methods = ["leiden"]
        resolutions = [0.6]
        n_neighbors = [30]
        best_params = {'resolution': 0, 'ari': 0, 'nmi': 0, 'method': None, 'n_neighbors': 0}

        for n_neighbor in n_neighbors:
            sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=n_neighbor)
            for method in clustering_methods:
                clustering_func = sc.tl.leiden if method == 'leiden' else sc.tl.louvain
                for resolution in resolutions:
                    clustering_func(adata, resolution=resolution, key_added=method)
                    ari = adjusted_rand_score(adata.obs['cell_type'], adata.obs[method])
                    nmi = normalized_mutual_info_score(adata.obs['cell_type'], adata.obs[method])

                    ari = round(ari, 4)
                    nmi = round(nmi, 4)

                    if ari > best_params['ari']:
                        best_params.update({
                            'resolution': resolution,
                            'ari': ari,
                            'method': method,
                            'n_neighbors': n_neighbor
                        })
                    if nmi > best_params['nmi']:
                        best_params['nmi'] = nmi

        return best_params

    def save_plots(self, adata: ad.AnnData, best_params: Dict[str, Any], title: str):
        """Save UMAP and t-SNE visualization plots."""
        sc.pp.neighbors(adata, use_rep='cell_embed', n_neighbors=best_params['n_neighbors'])

        # Generate and save UMAP plot
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=[best_params['method'], 'cell_type'], show=False)
        plt.title(f"{title} - ARI: {best_params['ari']}, NMI: {best_params['nmi']}")
        plt.savefig(f"{self.umap_dir}/{title}_Epoch_{self.epoch}_ARI_{best_params['ari']}}}.png")
        plt.close()

        # Generate and save t-SNE plot
        sc.tl.tsne(adata, use_rep='cell_embed')
        sc.pl.tsne(adata, color=[best_params['method'], 'cell_type'], show=False)
        plt.title(f"{title} - ARI: {best_params['ari']}, NMI: {best_params['nmi']}")
        plt.savefig(f"{self.tsne_dir}/{title}_Epoch_{self.epoch}_ARI_{best_params['ari']}}}.png")
        plt.close()

    
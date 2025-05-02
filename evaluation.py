import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import pybedtools
import pickle
import os
import random
from typing import Tuple, List, Optional, Dict, Any
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from sc_model import ScModel


class MLP(torch.nn.Module):
    """Simple MLP for GRN evaluation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def prepare_data(self, tf_row: torch.Tensor, positive_rows: torch.Tensor,
                     negative_rows: torch.Tensor) -> Tuple:
        """Prepare data for MLP training and evaluation."""
        # Create pairs by concatenating TF embedding with target embeddings
        tf_expanded = tf_row.repeat(positive_rows.size(0), 1)
        positive_pairs = torch.cat([tf_expanded, positive_rows], dim=1)

        tf_expanded = tf_row.repeat(negative_rows.size(0), 1)
        negative_pairs = torch.cat([tf_expanded, negative_rows], dim=1)

        # Create labels
        positive_labels = torch.ones(positive_rows.size(0), 1)
        negative_labels = torch.zeros(negative_rows.size(0), 1)

        # Combine positive and negative examples
        all_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Shuffle the data
        indices = torch.randperm(all_pairs.size(0))
        all_pairs = all_pairs[indices]
        all_labels = all_labels[indices]

        # Split into train and test sets (80/20)
        split_idx = int(all_pairs.size(0) * 0.8)
        train_pairs = all_pairs[:split_idx].to(self.device)
        test_pairs = all_pairs[split_idx:].to(self.device)
        train_labels = all_labels[:split_idx].to(self.device)
        test_labels = all_labels[split_idx:].to(self.device)

        return train_pairs, test_pairs, train_labels, test_labels

    def train_model(self, train_pairs: torch.Tensor, train_labels: torch.Tensor,
                    epochs: int = 50, lr: float = 0.001) -> None:
        """Train the MLP model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            outputs = self(train_pairs)
            loss = criterion(outputs, train_labels)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self, test_pairs: torch.Tensor, test_labels: torch.Tensor) -> Tuple:
        """Evaluate the MLP model."""
        self.eval()
        with torch.no_grad():
            outputs = self(test_pairs)
            predictions = (outputs >= 0.5).float()

            # Calculate accuracy
            accuracy = (predictions == test_labels).float().mean().item()

            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(test_labels.cpu().numpy(),
                                             outputs.cpu().numpy())
            roc_auc = auc(fpr, tpr)

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(test_labels.cpu().numpy(),
                                                          outputs.cpu().numpy())
            pr_auc = average_precision_score(test_labels.cpu().numpy(),
                                             outputs.cpu().numpy())

        return accuracy, fpr, tpr, roc_auc, precision, recall, pr_auc, outputs.cpu().numpy()


def evaluate_grn(model: ScModel, rna_adata_gnn: ad.AnnData, atac_adata_gnn: ad.AnnData,
                 grn_matrix_path: str, bed_file_path: str, plot_path: str,
                 method: str = 'dot'):
    """Evaluate Gene Regulatory Network predictions."""

    def check_overlap(df1: pd.DataFrame, df2: pd.DataFrame) -> List[int]:
        """Check for overlapping regions between two genomic coordinate dataframes."""
        overlaps = []
        for idx, row in df1.iterrows():
            condition = (
                    (df2['chrom'] == row['chromosome']) &
                    (df2['start'] <= row['end']) &
                    (df2['end'] >= row['start'])
            )
            if df2[condition].shape[0] > 0:
                overlaps.append(row['index'])
        return overlaps

    def intersect_and_sample(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        """Find intersection and sample non-intersecting elements."""
        set_a = set(a)
        set_b = set(b)
        intersection = list(set_a.intersection(set_b))
        non_intersecting_a = list(set_a.difference(set_b))

        if len(non_intersecting_a) <= len(b):
            return intersection, non_intersecting_a
        return intersection, random.sample(non_intersecting_a, len(b))

    # Load and process bed file
    bed_file = pybedtools.BedTool(bed_file_path)
    bed_df = pd.read_table(bed_file.fn, header=None,
                           names=['chrom', 'start', 'end', 'name', 'score'])

    # Load GRN matrix
    with open(grn_matrix_path, 'rb') as file:
        grn_matrix = pickle.load(file)

    # Extract TF name and cell type from bed file path
    base_name = os.path.basename(bed_file_path)
    components = base_name.split('_')
    TF_name = components[1]
    cell_type = components[2].split('.')[0]

    if TF_name not in rna_adata_gnn.var_names:
        print("TF not found in the gene identifiers.")
        return

    # Process GRN data
    TF_index = rna_adata_gnn.var_names.get_loc(TF_name)
    TF_row = grn_matrix[TF_index]
    ones_indices = TF_row.indices[TF_row.data == 1]
    adjusted_indices = ones_indices - rna_adata_gnn.shape[1]  # Adjust for RNA indices

    selected_peaks = atac_adata_gnn.var_names[adjusted_indices]

    # Prepare peak data
    connected_peak_data = {
        'index': [],
        'chromosome': [],
        'start': [],
        'end': []
    }

    for i, peak in enumerate(selected_peaks):
        chrom, positions = peak.split(':')
        start, end = positions.split('-')
        connected_peak_data['index'].append(ones_indices[i])
        connected_peak_data['chromosome'].append(chrom)
        connected_peak_data['start'].append(int(start))
        connected_peak_data['end'].append(int(end))

    connected_peak_df = pd.DataFrame(connected_peak_data)
    bed_df['start'] = bed_df['start'].astype(int)
    bed_df['end'] = bed_df['end'].astype(int)

    # Find overlaps and compute scores
    positive_indices = check_overlap(connected_peak_df, bed_df)
    _, negative_indices = intersect_and_sample(ones_indices.tolist(), positive_indices)
    compute_scores(model, TF_index, positive_indices, negative_indices, plot_path,
                   cell_type, TF_name, method)


def compute_scores(model: ScModel, TF_index: int, positive_indices: List[int],
                   negative_indices: List[int], plot_path: str, cell_type: str,
                   tf_name: str, method: str = 'dot'):
    """Compute scores for GRN evaluation using different methods."""
    tf_row = model.best_emb[TF_index].unsqueeze(0).cpu()
    positive_rows = model.best_emb[positive_indices].cpu()
    negative_rows = model.best_emb[negative_indices].cpu()

    if method in ['dot', 'finetune_dot']:
        scores, labels = _compute_dot_product_scores(tf_row, positive_rows,
                                                     negative_rows)
    elif method in ['pear', 'finetune_pear']:
        scores, labels = _compute_pearson_scores(tf_row, positive_rows,
                                                 negative_rows)
    elif method in ['mlp', 'mlp_finetune']:
        scores, labels = _compute_mlp_scores(model, tf_row, positive_rows,
                                             negative_rows)
    else:
        raise ValueError(f"Unknown method: {method}")

    _compute_and_plot_metrics(scores, labels, plot_path, cell_type,
                              tf_name, method)


def _compute_dot_product_scores(tf_row: torch.Tensor,
                                positive_rows: torch.Tensor,
                                negative_rows: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scores using dot product method."""
    positive_scores = torch.mm(positive_rows, tf_row.t())
    negative_scores = torch.mm(negative_rows, tf_row.t())

    positive_probs = torch.sigmoid(positive_scores)
    negative_probs = torch.sigmoid(negative_scores)

    labels = torch.cat([torch.ones(positive_probs.size(0)),
                        torch.zeros(negative_probs.size(0))])
    scores = torch.cat([positive_probs, negative_probs]).squeeze()

    return scores.detach().numpy(), labels.numpy()


def _compute_pearson_scores(tf_row: torch.Tensor,
                            positive_rows: torch.Tensor,
                            negative_rows: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scores using Pearson correlation."""
    tf_row_np = tf_row.squeeze().numpy()
    positive_corrs = np.array([pearsonr(tf_row_np, row.numpy())[0]
                               for row in positive_rows])
    negative_corrs = np.array([pearsonr(tf_row_np, row.numpy())[0]
                               for row in negative_rows])

    scores = np.concatenate([positive_corrs, negative_corrs])
    labels = np.concatenate([np.ones(len(positive_corrs)),
                             np.zeros(len(negative_corrs))])

    return scores, labels


def _compute_mlp_scores(model: ScModel, tf_row: torch.Tensor,
                        positive_rows: torch.Tensor,
                        negative_rows: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scores using MLP model."""
    mlp_model = MLP(model.best_emb.size(1), 128, 1)
    pairs_train, pairs_test, labels_train, labels_test = mlp_model.prepare_data(
        tf_row, positive_rows, negative_rows)

    mlp_model.train_model(pairs_train, labels_train, epochs=500, lr=0.0001)
    accuracy, fpr, tpr, roc_auc, precision, recall, pr_auc, scores = mlp_model.evaluate_model(pairs_test, labels_test)

    return scores, labels_test.cpu().numpy()


def _compute_and_plot_metrics(scores: np.ndarray, labels: np.ndarray,
                              plot_path: str, cell_type: str, tf_name: str,
                              method: str):
    """Compute and plot evaluation metrics."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    _plot_roc(fpr, tpr, roc_auc, plot_path, cell_type, tf_name, method)
    _plot_precision_recall(recall, precision, pr_auc, plot_path, cell_type,
                           tf_name, method)


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
              path: str, cell_type: str, tf_name: str, method: str):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{path}/{method}_{tf_name}_{cell_type}_roc.png')
    plt.close()


def _plot_precision_recall(recall: np.ndarray, precision: np.ndarray,
                           pr_auc: float, path: str, cell_type: str,
                           tf_name: str, method: str):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'{path}/{method}_{tf_name}_{cell_type}_prc.png')
    plt.close()
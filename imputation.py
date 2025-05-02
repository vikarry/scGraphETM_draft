import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Any
from scipy.stats import pearsonr, spearmanr

from sc_model import ScModel


def impute(model: ScModel, data_loader, impute_plot_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Impute RNA and ATAC data and calculate correlations across the entire dataset.

    Args:
        model: Trained ScModel
        data_loader: ScDataLoader containing the data to impute
        impute_plot_path: Optional path to save imputation plots

    Returns:
        Dictionary containing correlation metrics for both modalities
    """
    encoder1, encoder2, gnn, decoder1, decoder2 = model.models
    model.eval()

    # Initialize lists to store all predictions and ground truth
    all_predictions = {
        'RNA': {'original': [], 'imputed': []},
        'ATAC': {'original': [], 'imputed': []}
    }

    with torch.no_grad():
        for batch in tqdm(data_loader.test_loader, desc="Imputing"):
            # Move batch to device
            batch = [tensor.to(model.device) if isinstance(tensor, torch.Tensor)
                     else tensor for tensor in batch]

            # Unpack batch data
            (RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
             RNA_tensor_gnn, RNA_tensor_normalized_gnn,
             ATAC_tensor_gnn, ATAC_tensor_normalized_gnn) = batch

            # Get embeddings and topic distributions for both modalities
            # RNA -> ATAC
            mu1, log_sigma1, _ = encoder1(RNA_tensor_normalized)
            z1 = model.reparameterize(mu1, log_sigma1)
            theta1 = F.softmax(z1, dim=-1)

            # ATAC -> RNA
            mu2, log_sigma2, _ = encoder2(ATAC_tensor_normalized)
            z2 = model.reparameterize(mu2, log_sigma2)
            theta2 = F.softmax(z2, dim=-1)

            # Initialize variables for GNN
            rho, eta = None, None

            # GNN processing if enabled
            if model.use_gnn:
                edge_index = data_loader.train_loader.dataset.dataset.edge_index.to(model.device)

                if model.use_xtrimo:
                    gene_embeddings = model.gene_embedding(RNA_tensor)
                    atac_embeddings = model.atac_embedding(ATAC_tensor)
                    fm = torch.cat((gene_embeddings, atac_embeddings), dim=0)
                else:
                    specific_fm = torch.cat((ATAC_tensor_normalized.T, RNA_tensor_normalized.T), dim=0)
                    specific_fm = model.batch_to_emb(specific_fm).to(model.device)
                    fm = specific_fm * model.node2vec.to(model.device)

                emb = gnn(fm, edge_index)
                rho, eta = model.split_tensor(emb, RNA_tensor_gnn.shape[1])

            # Imputation
            # ATAC -> RNA
            rna_imputed = decoder1(theta2, rho if model.use_gnn else None)
            rna_imputed = torch.exp(rna_imputed) - 1

            # RNA -> ATAC
            atac_imputed = decoder2(theta1, eta if model.use_gnn else None)
            atac_imputed = torch.exp(atac_imputed) - 1

            # Store results
            all_predictions['RNA']['original'].append(RNA_tensor_normalized.cpu().numpy())
            all_predictions['RNA']['imputed'].append(rna_imputed.cpu().numpy())
            all_predictions['ATAC']['original'].append(ATAC_tensor_normalized.cpu().numpy())
            all_predictions['ATAC']['imputed'].append(atac_imputed.cpu().numpy())

    # Concatenate all results
    for modality in ['RNA', 'ATAC']:
        for data_type in ['original', 'imputed']:
            all_predictions[modality][data_type] = np.concatenate(
                all_predictions[modality][data_type], axis=0)

    # Calculate correlations
    correlations = {}
    for modality in ['RNA', 'ATAC']:
        orig = all_predictions[modality]['original']
        imp = all_predictions[modality]['imputed']

        pearson_corrs = []
        spearman_corrs = []

        print(f"\nCalculating correlations for {modality}...")
        for j in tqdm(range(orig.shape[0])):
            # Skip features with all zeros
            if np.all(orig[:, j] == 0) or np.all(imp[:, j] == 0):
                continue

            # Calculate correlations
            pear_corr, _ = pearsonr(orig[j, :], imp[j, :])
            spear_corr, _ = spearmanr(orig[j, :], imp[j, :])

            if not np.isnan(pear_corr):
                pearson_corrs.append(pear_corr)
            if not np.isnan(spear_corr):
                spearman_corrs.append(spear_corr)

        correlations[modality] = {
            'pearson': np.mean(pearson_corrs),
            'spearman': np.mean(spearman_corrs)
        }

        print(f"\n==== {modality} Imputation Results ====")
        print(f"Pearson correlation: {correlations[modality]['pearson']:.4f}")
        print(f"Spearman correlation: {correlations[modality]['spearman']:.4f}")

    # Plot results if path provided
    if impute_plot_path:
        _plot_imputation_results(
            all_predictions['RNA']['original'],
            all_predictions['RNA']['imputed'],
            all_predictions['ATAC']['original'],
            all_predictions['ATAC']['imputed'],
            impute_plot_path
        )

    return correlations


def _plot_imputation_results(rna_original: np.ndarray, rna_imputed: np.ndarray,
                             atac_original: np.ndarray, atac_imputed: np.ndarray,
                             plot_path: str):
    """
    Plot imputation results including scatterplots and correlation distributions.

    Args:
        rna_original: Original RNA data
        rna_imputed: Imputed RNA data
        atac_original: Original ATAC data
        atac_imputed: Imputed ATAC data
        plot_path: Path to save plots
    """
    # Set up plotting parameters
    plt.style.use('seaborn')
    fig_size = (10, 8)

    for modality, orig, imp in [('RNA', rna_original, rna_imputed),
                                ('ATAC', atac_original, atac_imputed)]:
        # Scatter plot of original vs imputed values
        plt.figure(figsize=fig_size)
        sample_size = 10000
        indices = np.random.choice(orig.size, sample_size, replace=False)
        plt.scatter(orig.flat[indices], imp.flat[indices], alpha=0.1, s=1)
        plt.xlabel('Original Values')
        plt.ylabel('Imputed Values')
        plt.title(f'{modality} Original vs Imputed Values')

        # Add correlation line
        z = np.polyfit(orig.flat[indices], imp.flat[indices], 1)
        p = np.poly1d(z)
        plt.plot(orig.flat[indices], p(orig.flat[indices]), "r--", alpha=0.8)

        corr = np.corrcoef(orig.flat[indices], imp.flat[indices])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.4f}',
                 transform=plt.gca().transAxes)

        plt.savefig(os.path.join(plot_path, f'{modality}_scatter.png'))
        plt.close()

        # Distribution of correlations across features
        correlations = []
        for j in range(orig.shape[1]):
            if not np.all(orig[:, j] == 0) and not np.all(imp[:, j] == 0):
                corr = pearsonr(orig[j, :], imp[j, :])
                if not np.isnan(corr):
                    correlations.append(corr)

        plt.figure(figsize=fig_size)
        plt.hist(correlations, bins=50)
        plt.xlabel('Correlation')
        plt.ylabel('Frequency')
        plt.title(f'{modality} Feature-wise Correlation Distribution')
        plt.axvline(np.mean(correlations), color='r', linestyle='dashed',
                    label=f'Mean: {np.mean(correlations):.4f}')
        plt.legend()
        plt.savefig(os.path.join(plot_path, f'{modality}_correlation_dist.png'))
        plt.close()
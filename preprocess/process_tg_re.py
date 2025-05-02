import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
from joblib import Parallel, delayed
import anndata as ad
import pickle
import scanpy as sc
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os
import argparse


def prepare_matrix(atac_adata):
    # Create a full list of unique peaks and map them to indices
    peak_indices = {peak: idx for idx, peak in enumerate(atac_adata.var_names)}
    return peak_indices


def process_gene_to_matrix(gene_idx, gene_name, gene_chrom, gene_start, atac_adata, distance, top, method,
                           peak_indices, rna_adata):
    connections = np.zeros(len(peak_indices))

    chrom_mask = atac_adata.var['chrom'] == gene_chrom
    same_chrom_peaks_adata = atac_adata[:, chrom_mask].copy()
    same_chrom_peaks_names = atac_adata.var[chrom_mask].index

    mse = 0
    std = 0

    # Filter peaks that are on the same chromosome and within the specified distance
    if method in ['nearby', 'both']:

        distances = np.abs(same_chrom_peaks_adata.var['chromStart'].astype(int) - gene_start)

        relevant_peaks_idx = distances[distances <= distance].index

        if len(relevant_peaks_idx) == 0:
            return connections, mse, std

        numerical_indices = [peak_indices[string_index] for string_index in relevant_peaks_idx]
        connections[numerical_indices] = 1

    if method in ['gbm', 'both']:
        if same_chrom_peaks_adata.shape[1] == 0:
            return connections, mse, std

        X = same_chrom_peaks_adata.X  # Peak expressions as features
        y = rna_adata.X[:, gene_idx]  # Gene expression as the target

        # Convert to dense arrays if necessary
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        y_dense = y.toarray() if hasattr(y, "toarray") else y

        try:
            X_train, X_val, y_train, y_val = train_test_split(X_dense, y_dense, test_size=0.2, random_state=42,
                                                              shuffle=True)

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.5,
                min_child_weight=10,
                random_state=42,
                tree_method='gpu_hist'  # Remove this line if not using GPU
            )

            # Fit the model
            model.set_params(early_stopping_rounds=10)
            model.fit(X_train, y_train.ravel(), eval_set=[(X_val, y_val.ravel())], verbose=False)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            std = np.std(y_pred)

            importances = model.feature_importances_
            if len(importances) > 0:
                top_peak_indices = np.argsort(importances)[-min(top, len(importances)):]

                top_original_names = [same_chrom_peaks_names[int_idx] for int_idx in top_peak_indices]
                top_numerical_indices = [peak_indices[string_index] for string_index in top_original_names]
                connections[top_numerical_indices] = 1
        except Exception as e:
            print(f"Error in GBM for gene {gene_name}: {e}")

    if gene_idx % 1000 == 0:
        print(f"Processed gene {gene_idx} with {connections.sum()} connections.")

    return connections, mse, std


def create_connection_matrix(rna_adata, atac_adata, distance=1e6, top=5, method='both', save_path=None):
    """
    Create a connection matrix between genes and peaks.

    Args:
        rna_adata: AnnData object for RNA-seq data
        atac_adata: AnnData object for ATAC-seq data
        distance: Distance threshold (in bp) for nearby method
        top: Number of top peaks to select for gbm method
        method: Method to use, one of 'nearby', 'gbm', or 'both'
        save_path: Path to save filtered ATAC-seq data

    Returns:
        Sparse matrix of gene-peak connections
    """
    peak_indices = prepare_matrix(atac_adata)
    gene_matrix = lil_matrix((rna_adata.shape[1], atac_adata.shape[1]), dtype=int)

    mse_list = []
    std_list = []

    for gene_idx, gene_name in tqdm(enumerate(rna_adata.var_names), total=len(rna_adata.var_names),
                                    desc="Processing Genes"):
        try:
            gene_chrom = rna_adata.var.loc[gene_name, 'chrom']
            gene_start = int(rna_adata.var.loc[gene_name, 'chromStart'])

            connections, gene_mse, gene_std = process_gene_to_matrix(
                gene_idx, gene_name, gene_chrom, gene_start,
                atac_adata, distance, top, method, peak_indices, rna_adata
            )

            gene_matrix[gene_idx, :] = connections
            mse_list.append(gene_mse)
            std_list.append(gene_std)
        except Exception as e:
            print(f"Error processing gene {gene_name} at index {gene_idx}: {e}")
            continue

    # Convert to CSR matrix for efficient operations
    gene_matrix_csr = csr_matrix(gene_matrix)

    # Find columns with non-zero values (peaks connected to at least one gene)
    non_zero_columns = np.unique(gene_matrix_csr.nonzero()[1])

    # Create a new matrix with only the non-zero columns
    if len(non_zero_columns) > 0:
        gene_matrix_filtered = gene_matrix_csr[:, non_zero_columns]

        # Create a filtered ATAC adata with only the connected peaks
        filtered_atac_adata = atac_adata[:, non_zero_columns].copy()

        # Save the filtered ATAC data
        if save_path:
            filtered_atac_adata.write(save_path)
            print(f"Number of peaks within {distance} bp: {filtered_atac_adata.shape}")
    else:
        print("Warning: No non-zero columns found in the gene matrix.")
        gene_matrix_filtered = gene_matrix_csr
        filtered_atac_adata = atac_adata.copy()

        if save_path:
            filtered_atac_adata.write(save_path)

    return gene_matrix_filtered


def parse_args():
    parser = argparse.ArgumentParser(description='Create TG-RE connections')
    parser.add_argument('--rna_path', required=True, help='Path to RNA-seq h5ad file')
    parser.add_argument('--atac_path', required=True, help='Path to ATAC-seq h5ad file')
    parser.add_argument('--flag', default='nearby', choices=['nearby', 'gbm', 'both'],
                        help='Method to use for TG-RE connections')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top peaks to select for gbm method')
    parser.add_argument('--distance', type=int, default=1000000,
                        help='Distance threshold (in bp) for nearby method')
    parser.add_argument('--output_dir', default='./results',
                        help='Directory to save results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract dataset name from RNA path
    dataset = os.path.basename(args.rna_path).split('_')[0]

    # Format distance string for filenames
    distance_str = f"{int(args.distance / 1e6)}m" if args.distance >= 1e6 else f"{args.distance}bp"

    # Define output paths
    save_path = os.path.join(args.output_dir, f"{dataset}_tg_re_{args.flag}_dist_{distance_str}_ATAC.h5ad")
    matrix_path = os.path.join(args.output_dir, f"{dataset}_tg_re_{args.flag}_dist_{distance_str}_matrix.pkl")

    print(f"Loading RNA data from {args.rna_path}")
    rna_adata = ad.read_h5ad(args.rna_path)
    print(rna_adata)
    print(f"RNA data shape: {rna_adata.shape}")

    print(f"Loading ATAC data from {args.atac_path}")
    atac_adata = ad.read_h5ad(args.atac_path)
    print(atac_adata)
    print(f"ATAC data shape: {atac_adata.shape}")

    # # Normalize data if needed
    # # Note: Comment out if data is already normalized
    # print("Normalizing RNA data...")
    # sc.pp.normalize_total(rna_adata, target_sum=1e4)
    # sc.pp.log1p(rna_adata)
    #
    # print("Normalizing ATAC data...")
    # sc.pp.normalize_total(atac_adata, target_sum=1e4)
    # sc.pp.log1p(atac_adata)

    print("Creating TG-RE connections...")
    gene_peak_matrix = create_connection_matrix(
        rna_adata,
        atac_adata,
        distance=args.distance,
        method=args.flag,
        top=args.top,
        save_path=save_path
    )

    print(f"Gene-peak matrix shape: {gene_peak_matrix.shape}")
    print(f"Number of edges in the matrix: {gene_peak_matrix.nnz}")

    print(f"Saving gene-peak matrix to {matrix_path}")
    with open(matrix_path, 'wb') as f:
        pickle.dump(gene_peak_matrix, f)

    print("Processing completed successfully!")
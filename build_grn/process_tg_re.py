import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
from joblib import Parallel, delayed
import anndata
import pickle
import scanpy as sc
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def prepare_matrix(atac_adata):
    # Create a full list of unique peaks and map them to indices
    peak_indices = {peak: idx for idx, peak in enumerate(atac_adata.var_names)}
    return peak_indices


def process_gene_to_matrix(gene_idx, gene_name, gene_chrom, gene_start, atac_adata, distance, top, method,
                           peak_indices):

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

        if relevant_peaks_idx.empty:
            return connections, mse, std

        numerical_indices = [peak_indices[string_index] for string_index in relevant_peaks_idx]
        connections[numerical_indices] = 1

    if method in ['gbm', 'both']:
        X = same_chrom_peaks_adata.X  # Peak expressions as features
        y = rna_adata.X[:, gene_idx]  # Gene expression as the target

        # Convert to dense arrays if necessary
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        y_dense = y.toarray() if hasattr(y, "toarray") else y

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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        std = np.std(y_pred)
        # print("Mean Squared Error:", mse)

        importances = model.feature_importances_
        top_peak_indices = np.argsort(importances)[-top:]

        top_original_names = [same_chrom_peaks_names[int_idx] for int_idx in top_peak_indices]
        top_numerical_indices = [peak_indices[string_index] for string_index in top_original_names]
        connections[top_numerical_indices] = 1

    if gene_idx % 1000 == 0:
        print(f"Processed gene {gene_idx} with {connections.sum()} connections.")

    return connections, mse, std


def create_connection_matrix(rna_adata, atac_adata, distance=1e6, top=5, method='both', save_path=None):
    peak_indices = prepare_matrix(atac_adata)
    gene_matrix = lil_matrix((rna_adata.shape[1], atac_adata.shape[1]), dtype=int)

    mse_list = []
    std_list = []

    for gene_idx, gene_name in tqdm(enumerate(rna_adata.var_names), total=len(rna_adata.var_names),
                                    desc="Processing Genes"):
        connections, gene_mse, gene_std = process_gene_to_matrix(
            gene_idx, gene_name, rna_adata.var.loc[gene_name, 'chrom'], rna_adata.var.loc[gene_name, 'chromStart'],
            atac_adata, distance, top, method, peak_indices
        )

        gene_matrix[gene_idx, :] = connections
        mse_list.append(gene_mse)
        std_list.append(gene_std)

    non_zero_columns = np.unique(gene_matrix.nonzero()[1])
    gene_matrix = gene_matrix[:, non_zero_columns]

    filtered_atac_adata = atac_adata[:, non_zero_columns]

    filtered_atac_adata.write(save_path)
    print(f"Number of peaks within {distance} bp ",  filtered_atac_adata.shape)

    return csr_matrix(gene_matrix)


# Usage example

atac_path = '../../data/10x-Multiome-Pbmc10k-ATAC.h5ad'
rna_path = '../../data/10x-Multiome-Pbmc10k-RNA.h5ad'

dataset = 'PBMC'

rna_path_hvg = f'../../data/{dataset}/PBMC_processed/PBMC_filtered_nolog_rna_count.h5ad' # HVG

# rna_path_hvg = f"../PBMC/PBMC_processed/PBMC_filtered_rna_count.h5ad"

rna_adata = anndata.read(rna_path)
print(rna_adata.var.columns)
rna_adata_hvg = anndata.read(rna_path_hvg)
atac_adata = anndata.read(atac_path)
print(atac_adata.var.columns)

common_genes = rna_adata.var_names.intersection(rna_adata_hvg.var_names)
rna_adata = rna_adata[:, common_genes].copy()

sc.pp.normalize_total(rna_adata, target_sum=1e4)
sc.pp.log1p(rna_adata)

sc.pp.normalize_total(atac_adata, target_sum=1e4)
sc.pp.log1p(atac_adata)

flag = 'nearby'
top = 10
distance = 1e6
distance_str = '1m'
save_path = f'../{dataset}/GRN_files/{dataset}_tg_re_{flag}_dist_{distance_str}_ATAC.h5ad'
gene_peak_matrix = create_connection_matrix(rna_adata, atac_adata, distance=distance, method=flag, top=top, save_path=save_path)

print(gene_peak_matrix.shape)
print("Number of edges in the matrix:", gene_peak_matrix.nnz)

with open(f'../{dataset}/GRN_files/{dataset}_tg_re_{flag}_dist_{distance_str}_matrix.pkl', 'wb') as f:
    pickle.dump(gene_peak_matrix, f)


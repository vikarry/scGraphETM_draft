import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from glob import glob
import os
import pickle
import anndata as ad
import scanpy as sc
import numpy as np
import pickle
import scipy.sparse as sp
import time
import pandas as pd
import pyarrow.feather as feather
import pyarrow.csv as csv
import pyarrow as pa
import ast
from intervaltree import Interval, IntervalTree
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, lil_matrix, bmat
from scipy.sparse import hstack, vstack
import argparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv


def calculate_overlap(row, region_start, region_end):
    # Determine the start and end of the overlap
    overlap_start = max(row['chromStart'], region_start)
    overlap_end = min(row['chromEnd'], region_end)

    # Check for actual overlap
    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        union_start = min(row['chromStart'], region_start)
        union_end = max(row['chromEnd'], region_end)
        union_length = union_end - union_start
        overlap_percentage = (overlap_length / union_length) * 100
        return overlap_percentage
    else:
        return 0


def match_cistarget(rna_adata, atac_adata, score_df, motif2tf_df, threshold=3):
    peak_loc_df = atac_adata.var[['chrom', 'chromStart', 'chromEnd']]
    regions = score_df.columns[1:]  # Skip 'motifs' column if it exists

    peak_to_index = {peak: idx for idx, peak in enumerate(atac_adata.var.index)}
    gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}

    overlapping_peak_indices = set()
    significant_regions = set()
    region_to_peaks = {}

    # Initialize a sparse matrix
    num_peaks = atac_adata.shape[1]
    num_genes = rna_adata.shape[1]

    count = 0
    for region in tqdm(regions, desc="Matching CisTarget regions"):
        if count > 10000:
            break
        try:
            region_chr, region_loc = region.split(':')
            if region_chr.startswith('chr'):
                region_chr = region_chr[3:]  # Remove 'chr' prefix if present
            region_start, region_end = map(int, region_loc.split('-'))
        except ValueError:
            continue

        # Find overlaps - peaks that contain the region
        overlaps1 = peak_loc_df[
            (peak_loc_df['chrom'] == region_chr) &
            (peak_loc_df['chromStart'] <= region_start) &
            (peak_loc_df['chromEnd'] >= region_end)
            ]

        # Find overlaps - regions that contain the peak
        overlaps2 = peak_loc_df[(peak_loc_df['chrom'] == region_chr) &
                                (peak_loc_df['chromStart'] >= region_start) &
                                (peak_loc_df['chromEnd'] <= region_end)]

        if not overlaps1.empty:
            # Check if any score exceeds threshold
            if any(score_df[region] > threshold):
                significant_regions.add(region)
                region_to_peaks[region] = overlaps1.index.tolist()

        if not overlaps2.empty:
            if any(score_df[region] > threshold):
                significant_regions.add(region)
                if region in region_to_peaks:
                    region_to_peaks[region].extend(overlaps2.index.tolist())
                else:
                    region_to_peaks[region] = overlaps2.index.tolist()

        count += 1

    # Filter score dataframe to only include significant regions
    filtered_score_df = score_df[['motifs'] + list(significant_regions)] if 'motifs' in score_df.columns else score_df[
        list(significant_regions)]

    return filtered_score_df, overlapping_peak_indices, region_to_peaks


def create_cistarget_matrix(rna_adata, atac_adata, region_to_peak, score_df, motif_to_TF_dict, threshold=3,
                            percentile=0):
    """
    Create a matrix of TF-RE connections based on CisTarget scores.

    Args:
        rna_adata: AnnData object for RNA data
        atac_adata: AnnData object for ATAC data (filtered by TG-RE)
        region_to_peak: Dictionary mapping regions to peaks
        score_df: DataFrame with CisTarget scores
        motif_to_TF_dict: DataFrame mapping motifs to TFs
        threshold: Score threshold
        percentile: Optional percentile threshold

    Returns:
        Sparse matrix of TF-RE connections
    """
    # Initialize the sparse matrix: rows are genes, columns are peaks
    num_genes = rna_adata.shape[1]
    num_peaks = atac_adata.shape[1]
    gene_peak_matrix = lil_matrix((num_genes, num_peaks), dtype=np.int8)

    # Create mappings for quick lookup
    peak_to_index = {peak: idx for idx, peak in enumerate(atac_adata.var.index)}
    gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}

    # Create mapping from motif ID to gene name
    motif_to_gene = {row['#motif_id']: row['gene_name'] for _, row in motif_to_TF_dict.iterrows()}

    # Check if 'motifs' column exists in score_df
    has_motifs_col = 'motifs' in score_df.columns

    # Pre-filter columns in score_df based on region_to_peak mapping
    if has_motifs_col:
        columns_to_keep = ['motifs'] + list(region_to_peak.keys())
    else:
        columns_to_keep = list(region_to_peak.keys())

    filtered_df = score_df[columns_to_keep]

    if percentile != 0:
        # Calculate percentile excluding the 'motifs' column if it exists
        score_columns = [col for col in filtered_df.columns if
                         col != 'motifs'] if has_motifs_col else filtered_df.columns
        all_scores = filtered_df[score_columns].values.flatten()
        score_percentile = np.percentile(all_scores, percentile)
        print(f"Score {percentile}th percentile: {score_percentile}")

    # Iterate over each region and scores of that region in score_df
    for region in tqdm(region_to_peak.keys(), desc='Processing regions'):
        if region not in filtered_df.columns:
            continue

        # Obtain the corresponding peaks for the region
        peak_identifiers = region_to_peak[region]

        # Check if we have valid peak identifiers
        if not peak_identifiers:
            continue

        for peak_identifier in peak_identifiers:
            if peak_identifier not in peak_to_index:
                continue

            peak_index = peak_to_index[peak_identifier]

            # Get scores for this region
            # If 'motifs' column exists, we need to find motif indices that have scores > threshold
            if has_motifs_col:
                # First, get data for all motifs for this region
                scores = filtered_df[region]
                motifs = filtered_df['motifs']

                # Identify significant motifs based on threshold and percentile criterion
                if percentile == 0:
                    significant_indices = scores[scores > threshold].index
                else:
                    significant_indices = scores[(scores > threshold) & (scores > score_percentile)].index

                # For each significant motif index
                for motif_index in significant_indices:
                    motif = motifs.iloc[motif_index]
                    if motif not in motif_to_gene:
                        continue

                    gene_name = motif_to_gene[motif]
                    if gene_name not in gene_to_index:
                        continue

                    gene_index = gene_to_index[gene_name]
                    gene_peak_matrix[gene_index, peak_index] = 1
            else:
                # Direct row-based access if 'motifs' column doesn't exist
                for motif_idx, score in enumerate(filtered_df[region]):
                    if score > threshold and (percentile == 0 or score > score_percentile):
                        motif = filtered_df.index[motif_idx]  # Assuming motif IDs are in the index
                        if motif not in motif_to_gene:
                            continue

                        gene_name = motif_to_gene[motif]
                        if gene_name not in gene_to_index:
                            continue

                        gene_index = gene_to_index[gene_name]
                        gene_peak_matrix[gene_index, peak_index] = 1

    gene_peak_matrix = gene_peak_matrix.tocsr()
    return gene_peak_matrix


def combine_matrices(rna_adata, atac_adata, A, B, path):
    """
    Combine TF-RE and TG-RE matrices and save as GRN.

    Args:
        rna_adata: AnnData object for RNA data
        atac_adata: AnnData object for ATAC data
        A: TF-RE matrix
        B: TG-RE matrix
        path: Path to save combined GRN matrix
    """
    print(f"Matrices shapes: A={A.shape}, B={B.shape}")

    # Ensure both matrices have the same number of columns
    if A.shape[1] != B.shape[1]:
        print(f"Warning: Matrix column dimensions don't match. A has {A.shape[1]} columns, B has {B.shape[1]} columns.")

        # If one matrix has more columns, we need to resize for compatibility
        if A.shape[1] < B.shape[1]:
            # Resize A to match B's column count
            A_resized = sp.csr_matrix((A.shape[0], B.shape[1]), dtype=A.dtype)
            A_resized[:, :A.shape[1]] = A
            A = A_resized
            print(f"Resized matrix A to shape {A.shape}")
        else:
            # Resize B to match A's column count
            B_resized = sp.csr_matrix((B.shape[0], A.shape[1]), dtype=B.dtype)
            B_resized[:, :B.shape[1]] = B
            B = B_resized
            print(f"Resized matrix B to shape {B.shape}")

    # Combine the matrices
    A_plus_B = A + B

    # Convert to binary matrix (values > 0 become 1)
    A_plus_B_binary = (A_plus_B > 0).astype(int)

    gene_num = rna_adata.shape[1]
    peak_num = atac_adata.shape[1]

    # Create zero blocks for the diagonal parts of the matrix
    zero_block_gene = sp.csr_matrix((gene_num, gene_num))
    zero_block_peak = sp.csr_matrix((peak_num, peak_num))

    # Create the full GRN matrix with block structure:
    # [   0    | A_plus_B ]
    # [A_plus_B^T |   0   ]
    larger_matrix = sp.bmat([[zero_block_gene, A_plus_B_binary],
                             [A_plus_B_binary.T, zero_block_peak]], format='csr')

    num_edges = larger_matrix.count_nonzero()
    print(f"GRN Matrix shape: {larger_matrix.shape}")
    print(f"GRN Edges: {num_edges}")

    with open(path, 'wb') as f:
        pickle.dump(larger_matrix, f)

    print(f"GRN matrix saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Create TF-RE connections using CisTarget data')
    parser.add_argument('--rna_path', required=True, help='Path to RNA-seq h5ad file')
    parser.add_argument('--atac_path', required=True, help='Path to filtered ATAC-seq h5ad file (from TG-RE)')
    parser.add_argument('--cistarget_score', required=True, help='Path to CisTarget score file (.feather)')
    parser.add_argument('--motif2tf', required=True, help='Path to motif-to-TF mapping file (.tbl)')
    parser.add_argument('--tg_re_matrix', required=True, help='Path to TG-RE matrix (.pkl)')
    parser.add_argument('--threshold', type=int, default=3, help='Score threshold')
    parser.add_argument('--percentile', type=int, default=0, help='Score percentile threshold (0 to disable)')
    parser.add_argument('--output_dir', default='./results', help='Directory to save results')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract dataset name from RNA path
    dataset = os.path.basename(args.rna_path).split('_')[0]

    # Define output paths
    tf_re_matrix_path = os.path.join(args.output_dir, f"{dataset}_TF_RE_matrix_threshold_{args.threshold}.pkl")
    combined_grn_path = os.path.join(args.output_dir, f"{dataset}_combined_threshold{args.threshold}_GRN.pkl")

    print(f"Loading RNA data from {args.rna_path}")
    rna_adata = ad.read_h5ad(args.rna_path)
    print(f"RNA data shape: {rna_adata.shape}")

    print(f"Loading ATAC data from {args.atac_path}")
    atac_adata = ad.read_h5ad(args.atac_path)
    print(f"ATAC data shape: {atac_adata.shape}")

    print(f"Loading CisTarget score data from {args.cistarget_score}")
    score_df = feather.read_feather(args.cistarget_score)
    print(f"Score data shape: {score_df.shape}")

    print(f"Loading motif-to-TF mapping from {args.motif2tf}")
    motif2tf_df = pd.read_csv(args.motif2tf, delimiter='\t')
    if not all(col in motif2tf_df.columns for col in ['#motif_id', 'gene_name']):
        print("Warning: motif2tf file does not contain required columns '#motif_id' and 'gene_name'")
        print(f"Available columns: {motif2tf_df.columns.tolist()}")
    motif2tf_df = motif2tf_df.loc[:, ['#motif_id', 'motif_name', 'gene_name']]

    # Check for matching genes in RNA data and motif2tf mapping
    gene_col = 'gene_name' if 'gene_name' in rna_adata.var.columns else None
    if gene_col:
        matches = rna_adata.var[gene_col].isin(motif2tf_df['gene_name'].values)
    else:
        matches = rna_adata.var_names.isin(motif2tf_df['gene_name'].values)

    num_matches = sum(matches)
    print(f"Number of matching TFs in RNA data: {num_matches}")

    print("Matching CisTarget regions to ATAC peaks...")
    filtered_score_df, overlapping_peaks, region_to_peaks_df = match_cistarget(
        rna_adata, atac_adata, score_df, motif2tf_df, threshold=args.threshold
    )
    print(f"Number of matched regions: {len(region_to_peaks_df)}")

    print("Creating TF-RE matrix...")
    tf_re_matrix = create_cistarget_matrix(
        rna_adata, atac_adata, region_to_peaks_df, score_df, motif2tf_df,
        threshold=args.threshold, percentile=args.percentile
    )
    print(f"TF-RE matrix shape: {tf_re_matrix.shape}")
    print(f"Number of TF-RE edges: {tf_re_matrix.count_nonzero()}")

    print(f"Saving TF-RE matrix to {tf_re_matrix_path}")
    with open(tf_re_matrix_path, 'wb') as f:
        pickle.dump(tf_re_matrix, f)

    print(f"Loading TG-RE matrix from {args.tg_re_matrix}")
    with open(args.tg_re_matrix, 'rb') as f:
        tg_re_matrix = pickle.load(f)
    print(f"TG-RE matrix shape: {tg_re_matrix.shape}")
    print(f"Number of TG-RE edges: {tg_re_matrix.count_nonzero()}")

    print("Combining TF-RE and TG-RE matrices...")
    combine_matrices(
        rna_adata, atac_adata, tf_re_matrix, tg_re_matrix, combined_grn_path
    )

    print("CisTarget TF-RE processing completed successfully!")
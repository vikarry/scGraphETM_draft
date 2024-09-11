import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from glob import glob
import os
import pickle
import anndata
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

def match_cistarget(rna_adata, peak_adata, score_df, motif2tf_df, threshold=3):
    peak_loc_df = peak_adata.var[['chrom', 'chromStart', 'chromEnd']]
    regions = score_df.columns

    peak_to_index = {peak: idx for idx, peak in enumerate(atac_adata.var.index)}
    gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}

    overlapping_peak_indices = set()
    significant_regions = set()
    region_to_peaks = {}

    # Initialize a sparse matrix
    num_peaks = peak_adata.shape[1]
    num_genes = rna_adata.shape[1]

    count = 0
    for region in tqdm(regions):
        if count > 10000:
            break
        try:
            region_chr, region_loc = region.split(':')
            region_chr = region_chr[3:]
            region_start, region_end = map(int, region_loc.split('-'))
        except ValueError:
            continue

        # for index, row in peak_loc_df.iterrows():
        #     print(f"Peak on {row['chrom']} from {row['chromStart']} to {row['chromEnd']}")
        #     print(f"Region on {region_chr} from {region_start} to {region_end}")
        #     print("---")

            # Find overlaps
        overlaps1 = peak_loc_df[
            (peak_loc_df['chrom'] == region_chr) &
            (peak_loc_df['chromStart'] <= region_start) &
            (peak_loc_df['chromEnd'] >= region_end)
            ]

        overlaps2 = peak_loc_df[(peak_loc_df['chrom'] == region_chr) &
            (peak_loc_df['chromStart'] >= region_start) &
            (peak_loc_df['chromEnd'] <= region_end)]

        # peak_loc_df['overlap_percentage'] = peak_loc_df.apply(
        #     lambda row: calculate_overlap(row, region_start, region_end),
        #     axis=1
        # )
        #
        # min_overlap_percentage = 80
        #
        # # Filter peaks based on the overlap percentage
        # overlaps = peak_loc_df[
        #     (peak_loc_df['chrom'] == region_chr) &
        #     (peak_loc_df['overlap_percentage'] >= min_overlap_percentage)
        #     ]

        if not overlaps1.empty:
            # print(region, overlaps.index.tolist())
            if any(score_df[region] > threshold):
                # overlapping_peak_indices.update(overlaps.index.tolist())
                significant_regions.add(region)
                region_to_peaks[region] = overlaps1.index

        if not overlaps2.empty:
            if any(score_df[region] > threshold):
                significant_regions.add(region)
                region_to_peaks[region] = overlaps2.index

        # count += 1

    filtered_score_df = score_df[significant_regions]

    return filtered_score_df, overlapping_peak_indices, region_to_peaks


def create_cistarget_matrix(rna_adata, new_atac_adata, region_to_peak, score_df, motif_to_TF_dict, threshold=3,
                            percentile=0):
    # Initialize the sparse matrix: rows are genes, columns are peaks
    num_genes = rna_adata.shape[1]
    num_peaks = new_atac_adata.shape[1]
    gene_peak_matrix = lil_matrix((num_genes, num_peaks), dtype=np.int8)

    # Create mappings for quick lookup
    peak_to_index = {peak: idx for idx, peak in enumerate(new_atac_adata.var.index)}
    gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}

    motif_to_gene = {row['#motif_id']: row['gene_name'] for _, row in motif_to_TF_dict.iterrows()}

    # Pre-filter columns in score_df based on region_to_peak mapping
    columns_to_keep = region_to_peak.keys()
    filtered_df = score_df.loc[:, score_df.columns.isin(columns_to_keep)]

    if percentile != 0:
        all_scores = filtered_df.values.flatten()
        score_percentile = np.percentile(all_scores, percentile)
        # print(score_percentile)

    # Iterate over each region and scores of that region in score_df
    for region, scores in tqdm(filtered_df.items(), desc='Processing regions', total=filtered_df.shape[1]):
        # print(region, scores)

        # Obtain the corresponding peak to the region
        peak_identifier = region_to_peak[region]

        peak_index = peak_to_index[str(peak_identifier[0])]

        # Identify significant motifs based on threshold and 95th percentile criterion
        if percentile == 0:
            significant_motifs = scores[(scores > threshold)].index
        else:
            significant_motifs = scores[(scores > threshold) & (scores > score_percentile)].index
        # print(len(significant_motifs))
        for motif_index in significant_motifs:
            motif = score_df.loc[motif_index, 'motifs']
            if motif not in motif_to_gene:
                continue

            gene_name = motif_to_gene[motif]
            if gene_name not in gene_to_index:
                continue

            gene_index = gene_to_index[gene_name]
            gene_peak_matrix[gene_index, peak_index] = 1

    gene_peak_matrix = gene_peak_matrix.tocsr()
    return gene_peak_matrix


def combine_matrices(rna_adata, peak_adata, A, B, path):
    print(A.shape, B.shape)
    A_plus_B = A + B

    A_plus_B_binary = (A_plus_B > 0).astype(int)
    gene_num = rna_adata.shape[1]
    peak_num = peak_adata.shape[1]

    zero_block_gene = csr_matrix((gene_num, gene_num))
    zero_block_peak = csr_matrix((peak_num, peak_num))

    larger_matrix = bmat([[zero_block_gene, A_plus_B_binary], [A_plus_B_binary.T, zero_block_peak]], format='csr')
    num_edges = larger_matrix.count_nonzero()
    print("GRN Edges: ", num_edges)
    with open(path, 'wb') as f:
        pickle.dump(larger_matrix, f)

if __name__ == '__main__':

    dataset = "PBMC"
    tissue = "PBMC"

    threshold = 3
    threshold_value = "threshold3"
    threshold_title = "threshold3"

    atac_path = f'../{dataset}/GRN_files/{dataset}_tg_re_nearby_dist_1m_ATAC.h5ad'
    rna_path = f"../{dataset}/{tissue}_hvg_rna_count.h5ad"

    data_score = "../hg38_screen_v10_clust.regions_vs_motifs.scores.feather"
    cistarget_motif2tf = '../motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'

    rna_adata = anndata.read(rna_path)
    atac_adata = anndata.read(atac_path)

    print(rna_adata.shape)

    score_df = feather.read_feather(data_score)
    motif2tf_df = pd.read_csv(cistarget_motif2tf, delimiter='\t')
    motif2tf_df = motif2tf_df.loc[:, ['#motif_id', 'motif_name', 'gene_name']]

    matches = rna_adata.var['gene_name'].isin(motif2tf_df['gene_name'].values)
    num_matches = matches.sum()
    print(f"Number of matching motifs: {num_matches}")

    filtered_score_df, overlapping_peaks, region_to_peaks_df = match_cistarget(rna_adata, atac_adata, score_df,
                                                                               motif2tf_df, threshold=threshold)
    # print(region_to_peaks_df)

    tf_re_matrix = create_cistarget_matrix(rna_adata, atac_adata, region_to_peaks_df, score_df, motif2tf_df, threshold=threshold)
    with open(f'../{dataset}/GRN_files/TF_RE_hvg_matrix_threshold_{threshold}.pkl', 'wb') as f:
        pickle.dump(tf_re_matrix, f)

    num_edges = tf_re_matrix.count_nonzero()
    print("Edges: ", num_edges)

    with open(f'../{dataset}/GRN_files/{dataset}_tg_re_nearby_dist_1m_matrix.pkl', 'rb') as f:
        tg_re_matrix = pickle.load(f)

    combine_matrices(rna_adata, atac_adata, tf_re_matrix, tg_re_matrix,
                     f'../{dataset}/GRN_files/{tissue}_1M_{threshold_title}_GRN.pkl')


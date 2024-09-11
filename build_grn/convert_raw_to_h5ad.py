import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import pybedtools
import pyreadr

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import scipy.io
from scipy.sparse import csr_matrix

"""
Process count_matrix.RDS using R first into .mtx format

getwd()
setwd(*** Path/to/file/ ***)
count_matrix <- readRDS("GSE204682_count_matrix.RDS")

Matrix::writeMM(count_matrix, "count_matrix.mtx")

"""


def process_atac_raw_data(count_matrix_path, bardcodes_path, bed_file_path, atac_save_path):
    matrix = scipy.io.mmread(count_matrix_path)
    sparse_matrix = csr_matrix(matrix)

    obs_df = pd.read_csv(bardcodes_path, sep='\t', index_col=0)
    bed_df = pd.read_csv(bed_file_path, header=None, sep='\t',
                         names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    bed_df = bed_df[['chrom', 'start', 'end', 'name']]
    # bed_df['chrom'] = 'chr' + bed_df['chrom'].astype(str)
    bed_df.rename(columns={
        'start': 'chromStart',
        'end': 'chromEnd'
    }, inplace=True)

    atac_adata = ad.AnnData(X=sparse_matrix.T, obs=obs_df, var=bed_df)
    atac_adata.obs_names_make_unique()

    atac_adata.obs.index = atac_adata.obs_names

    atac_adata.var['region_index'] = atac_adata.var.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}",
                                                          axis=1)
    atac_adata.var_names = atac_adata.var['region_index']
    atac_adata.var.index = atac_adata.var['region_index']

    # print(atac_adata.var.head())
    atac_adata.write_h5ad(atac_save_path)


    # print(atac_adata.obs_names)
    # print(atac_adata.var_names)


def process_rna_raw_data(h5ad_file_path, ensemble_to_chrom_path, rna_save_path, count_matrix_path=None, barcodes_path=None,):
    if not count_matrix_path is None:
        matrix = scipy.io.mmread(count_matrix_path)
        sparse_matrix = csr_matrix(matrix)

        obs_df = pd.read_csv(bardcodes_path, sep='\t', index_col=0)

        rna_adata = ad.AnnData(X=sparse_matrix.T, obs=obs_df)
        rna_adata.obs_names_make_unique()
    else:
        rna_adata = ad.read_h5ad(h5ad_file_path)
        print(rna_adata.var['feature_name'])
        ensemble_to_chrom_csv = pd.read_csv(ensemble_to_chrom_path)
        # Prepend 'chrom' to each value in the 'Chromosome' column
        ensemble_to_chrom_csv['chromosome_name'] = 'chr' + ensemble_to_chrom_csv['chromosome_name'].astype(str)

        ensemble_to_chrom_csv.rename(columns={
            'chromosome_name': 'chrom',
            'start_position': 'chromStart',
            'end_position': 'chromEnd'
        }, inplace=True)

        ensemble_to_chrom_csv.set_index('ensembl_gene_id', inplace=True)
        rna_adata.var = rna_adata.var.join(ensemble_to_chrom_csv, how='left')
        # print(rna_adata.var.head())
        rna_adata.obs_names_make_unique()
        rna_adata.write_h5ad(rna_save_path)

        # print(rna_data.var.index)

# print(rna_data.obs['donor_id'].unique())
# rna_data_hvg = ad.read_h5ad('cerebral_cortex_rna_hvg.h5ad')
# print(rna_data_hvg)

# calculate hvg
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
#
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
#
# sc.pp.highly_variable_genes(adata)
# hvg_adata = adata[:, adata.var['highly_variable']].copy()
#
# print(hvg_adata)
# # To save
# hvg_adata.write_h5ad('cerebral_cortex_rna_hvg.h5ad')

if __name__ == '__main__':
    count_matrix_path = 'count_matrix.mtx'
    barcodes_path = 'GSE204682_barcodes.tsv'
    bed_file_path = 'GSE204682_peaks.bed'
    atac_save_path = 'cerebral_cortex_processed_atac.h5ad'

    # process_atac_raw_data(count_matrix_path, barcodes_path, bed_file_path, atac_save_path)

    rna_path = 'cerebral_cortex_rna.h5ad'
    ensemble_to_chrom_path = '../all_gene_chrom_positions.csv'
    rna_save_path = 'cerebral_cortex_processed_rna.h5ad'

    process_rna_raw_data(rna_path, ensemble_to_chrom_path, rna_save_path)

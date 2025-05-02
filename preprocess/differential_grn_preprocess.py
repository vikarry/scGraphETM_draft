import os
import gzip
import pandas as pd
import numpy as np
import anndata as ad
import scipy.io
from scipy.sparse import csr_matrix


def convert_rna_to_h5ad(features_file, barcodes_file, matrix_file, output_file):
    """
    Convert RNA-seq data files (features, barcodes, matrix) to h5ad format.

    Parameters:
    -----------
    features_file : str
        Path to the features/genes file (.tsv.gz)
    barcodes_file : str
        Path to the barcodes file (.tsv.gz)
    matrix_file : str
        Path to the matrix file (.mtx.gz)
    output_file : str
        Path to save the output h5ad file
    """
    print("Reading features...")
    # Read features file
    features_df = pd.read_csv(features_file, sep='\t', header=None, compression='gzip')

    # Standard 10X Genomics features file has 3 columns: gene_id, gene_name, feature_type
    if features_df.shape[1] >= 3:
        features_df.columns = ['gene_id', 'gene_name', 'feature_type'] + list(features_df.columns[3:])
    else:
        # If fewer columns, adjust accordingly
        if features_df.shape[1] == 2:
            features_df.columns = ['gene_id', 'gene_name']
        else:
            features_df.columns = ['gene_id']
            features_df['gene_name'] = features_df['gene_id']  # Use ID as name if no name provided

    print("Reading barcodes...")
    # Read barcodes file
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = [line.strip() for line in f]

    print("Reading matrix...")
    # Read matrix file - this is in Matrix Market format
    matrix = scipy.io.mmread(gzip.open(matrix_file, 'rb'))
    sparse_matrix = csr_matrix(matrix)

    # Create AnnData object
    # Note: The matrix from the mtx file is usually features x cells (genes x cells),
    # but AnnData expects cells x features, so we transpose it
    adata = ad.AnnData(X=sparse_matrix.T)

    # Add barcodes as observations
    adata.obs_names = barcodes
    adata.obs.index = adata.obs_names

    # Add gene information as variables
    adata.var = features_df
    adata.var_names = features_df['gene_id']
    adata.var.index = adata.var_names

    print(f"Created AnnData object with {adata.n_obs} cells and {adata.n_vars} genes")

    # Save as h5ad
    print(f"Saving to {output_file}...")
    adata.write_h5ad(output_file)
    print("Done!")

    return adata


def convert_atac_to_h5ad(barcodes_file, matrix_file, peaks_file, output_file):
    """
    Convert ATAC-seq data files (barcodes, matrix, peaks) to h5ad format.

    Parameters:
    -----------
    barcodes_file : str
        Path to the barcodes file (.tsv.gz)
    matrix_file : str
        Path to the matrix file (.mtx.gz)
    peaks_file : str
        Path to the peaks file (.bed.gz)
    output_file : str
        Path to save the output h5ad file
    """
    print("Reading barcodes...")
    # Read barcodes file
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = [line.strip() for line in f]

    print("Reading matrix...")
    # Read matrix file - this is in Matrix Market format
    matrix = scipy.io.mmread(gzip.open(matrix_file, 'rb'))
    sparse_matrix = csr_matrix(matrix)

    print("Reading peaks...")
    # Read peaks file (BED format)
    peaks_df = pd.read_csv(peaks_file, sep='\t', header=None,
                           compression='gzip',
                           names=['chrom', 'start', 'end', 'name', 'score', 'strand'])

    # Process peak data
    peaks_df = peaks_df[['chrom', 'start', 'end', 'name']]

    # Rename columns to standard names used in genomic data
    peaks_df.rename(columns={
        'start': 'chromStart',
        'end': 'chromEnd'
    }, inplace=True)

    # Create region names
    peaks_df['region_index'] = peaks_df.apply(
        lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1
    )

    # Create AnnData object
    # Note: The matrix from the mtx file is usually features x cells,
    # but AnnData expects cells x features, so we transpose it
    adata = ad.AnnData(X=sparse_matrix.T)

    # Add barcodes as observations
    adata.obs_names = barcodes
    adata.obs.index = adata.obs_names

    # Add peaks as variables
    adata.var = peaks_df
    adata.var_names = peaks_df['region_index']
    adata.var.index = adata.var_names

    print(f"Created AnnData object with {adata.n_obs} cells and {adata.n_vars} peaks")

    # Save as h5ad
    print(f"Saving to {output_file}...")
    adata.write_h5ad(output_file)
    print("Done!")

    return adata


def process_data_directory(data_dir="./data", output_dir="./data/processed"):
    """
    Process all RNA-seq and ATAC-seq data files in the data directory.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the data files
    output_dir : str
        Path to the directory to save the processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the data directory
    files = os.listdir(data_dir)

    # Group files by prefix
    file_groups = {}
    for file in files:
        if file.endswith('.tsv.gz') or file.endswith('.mtx.gz') or file.endswith('.bed.gz'):
            # Extract prefix (everything before the first underscore)
            prefix = file.split('_')[0]
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(file)

    # Process each group
    for prefix, group_files in file_groups.items():
        print(f"Processing files with prefix {prefix}...")

        # Find RNA-seq files
        rna_features = None
        rna_barcodes = None
        rna_matrix = None

        # Find ATAC-seq files
        atac_barcodes = None
        atac_matrix = None
        atac_peaks = None

        for file in group_files:
            full_path = os.path.join(data_dir, file)

            # RNA-seq files
            if 'rna_features' in file:
                rna_features = full_path
            elif 'rna_filtered_barcodes' in file:
                rna_barcodes = full_path
            elif 'rna_filtered_matrix' in file:
                rna_matrix = full_path

            # ATAC-seq files
            elif 'atac_filtered_barcodes' in file:
                atac_barcodes = full_path
            elif 'atac_filtered_matrix' in file:
                atac_matrix = full_path
            elif 'atac_peaks' in file:
                atac_peaks = full_path

        # Process RNA-seq files if all are found
        if rna_features and rna_barcodes and rna_matrix:
            rna_output = os.path.join(output_dir, f"{prefix}_rna.h5ad")
            print(f"Converting RNA-seq files to {rna_output}...")
            convert_rna_to_h5ad(rna_features, rna_barcodes, rna_matrix, rna_output)

        # Process ATAC-seq files if all are found
        if atac_barcodes and atac_matrix and atac_peaks:
            atac_output = os.path.join(output_dir, f"{prefix}_atac.h5ad")
            print(f"Converting ATAC-seq files to {atac_output}...")
            convert_atac_to_h5ad(atac_barcodes, atac_matrix, atac_peaks, atac_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert RNA-seq or ATAC-seq data to h5ad format')
    parser.add_argument('--data_dir', default='./data', help='Directory containing the data files')
    parser.add_argument('--output_dir', default='./data/processed', help='Directory to save the processed files')

    # Individual file processing
    parser.add_argument('--mode', choices=['rna', 'atac', 'auto'], default='auto',
                        help='Data type: rna for RNA-seq, atac for ATAC-seq, auto for automatic detection')
    parser.add_argument('--features', help='Path to features/genes file (.tsv.gz) - RNA-seq only')
    parser.add_argument('--peaks', help='Path to peaks file (.bed.gz) - ATAC-seq only')
    parser.add_argument('--barcodes', help='Path to barcodes file (.tsv.gz)')
    parser.add_argument('--matrix', help='Path to matrix file (.mtx.gz)')
    parser.add_argument('--output', help='Path to save output h5ad file')

    args = parser.parse_args()

    # If individual files are specified, process them
    if args.barcodes and args.matrix and (args.features or args.peaks):
        if args.mode == 'rna' or (args.mode == 'auto' and args.features):
            if not args.output:
                args.output = os.path.join(args.output_dir, 'rna.h5ad')
            convert_rna_to_h5ad(args.features, args.barcodes, args.matrix, args.output)
        elif args.mode == 'atac' or (args.mode == 'auto' and args.peaks):
            if not args.output:
                args.output = os.path.join(args.output_dir, 'atac.h5ad')
            convert_atac_to_h5ad(args.barcodes, args.matrix, args.peaks, args.output)
    else:
        # Otherwise, process all files in the data directory
        process_data_directory(args.data_dir, args.output_dir)
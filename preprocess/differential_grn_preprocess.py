import os
import gzip
import pandas as pd
import numpy as np
import anndata as ad
import scipy.io
from scipy.sparse import csr_matrix


def convert_rna_to_h5ad(features_file, barcodes_file, matrix_file, output_file,
                        gene_pos_file=None, name2id_file=None):
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
    gene_pos_file : str, optional
        Path to gene positions CSV file containing chromosome and TSS info
    name2id_file : str, optional
        Path to gene name to Ensembl ID mapping CSV file
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

    # Process additional gene information if provided
    if gene_pos_file and os.path.exists(gene_pos_file):
        print(f"Adding gene position information from {gene_pos_file}...")
        gene_pos_df = pd.read_csv(gene_pos_file)

        # Ensure column names are standardized
        if 'ensembl_gene_id' in gene_pos_df.columns:
            gene_pos_df.set_index('ensembl_gene_id', inplace=True)
        elif 'gene_id' in gene_pos_df.columns:
            gene_pos_df.set_index('gene_id', inplace=True)

        # Standardize chromosome column names
        if 'chromosome_name' in gene_pos_df.columns:
            # Prepend 'chr' to chromosome names if not already present
            if not gene_pos_df['chromosome_name'].str.startswith('chr').all():
                gene_pos_df['chrom'] = 'chr' + gene_pos_df['chromosome_name'].astype(str)
            else:
                gene_pos_df['chrom'] = gene_pos_df['chromosome_name']

        # Standardize position column names
        position_mapping = {
            'start_position': 'chromStart',
            'end_position': 'chromEnd',
            'transcription_start_site': 'tss'
        }

        for old_col, new_col in position_mapping.items():
            if old_col in gene_pos_df.columns:
                gene_pos_df[new_col] = gene_pos_df[old_col]

        # Add TSS if not present but start_position is available
        if 'tss' not in gene_pos_df.columns and 'chromStart' in gene_pos_df.columns:
            gene_pos_df['tss'] = gene_pos_df['chromStart']

        # Join gene position data to adata.var
        adata.var = adata.var.join(gene_pos_df, how='left')

    # Process name to ID mapping if provided
    if name2id_file and os.path.exists(name2id_file):
        print(f"Adding gene name to ID mapping from {name2id_file}...")
        name2id_df = pd.read_csv(name2id_file)

        # Ensure column names are standardized
        if all(col in name2id_df.columns for col in ['gene_id', 'gene_name']):
            name2id_df.set_index('gene_id', inplace=True)

            # Fill in missing gene names if possible
            missing_names = adata.var['gene_name'].isna()
            if missing_names.any():
                for idx in adata.var.index[missing_names]:
                    if idx in name2id_df.index:
                        adata.var.at[idx, 'gene_name'] = name2id_df.at[idx, 'gene_name']

            # Add any additional columns from name2id mapping
            for col in name2id_df.columns:
                if col not in adata.var.columns:
                    adata.var = adata.var.join(name2id_df[[col]], how='left')

    # Annotate standard chromosomes 1-22, X, Y
    if 'chrom' in adata.var.columns:
        print("Adding chromosome annotation 1-22, X, Y...")
        valid_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

        # Handle chromosome naming with or without 'chr' prefix
        if not any(str(chrom).startswith('chr') for chrom in adata.var['chrom'] if isinstance(chrom, str)):
            valid_chroms = [c.replace('chr', '') for c in valid_chroms]

        # Add a column indicating if the gene is on standard chromosomes
        adata.var['standard_chromosome'] = adata.var['chrom'].isin(valid_chroms)

        # Count genes on standard chromosomes
        std_chrom_count = adata.var['standard_chromosome'].sum()
        total_count = adata.shape[1]
        print(f"{std_chrom_count} out of {total_count} genes are on standard chromosomes (1-22, X, Y)")

    # Store the data type in the anndata object
    adata.uns['data_type'] = 'rna'

    print(f"Final RNA AnnData object has {adata.n_obs} cells and {adata.n_vars} genes")

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

    # Annotate standard chromosomes 1-22, X, Y
    print("Adding chromosome annotation 1-22, X, Y...")
    valid_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

    # Handle chromosome naming with or without 'chr' prefix
    if not any(str(chrom).startswith('chr') for chrom in peaks_df['chrom'] if isinstance(chrom, str)):
        valid_chroms = [c.replace('chr', '') for c in valid_chroms]

    # Add a column indicating if the peak is on standard chromosomes
    peaks_df['standard_chromosome'] = peaks_df['chrom'].isin(valid_chroms)

    # Count peaks on standard chromosomes
    std_chrom_count = peaks_df['standard_chromosome'].sum()
    total_count = len(peaks_df)
    print(f"{std_chrom_count} out of {total_count} peaks are on standard chromosomes (1-22, X, Y)")

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

    # Store the data type in the anndata object
    adata.uns['data_type'] = 'atac'

    print(f"Final ATAC AnnData object has {adata.n_obs} cells and {adata.n_vars} peaks")

    # Save as h5ad
    print(f"Saving to {output_file}...")
    adata.write_h5ad(output_file)
    print("Done!")

    return adata


def standardize_barcodes(barcodes, strip_suffix=True):
    """
    Standardize cell barcodes to facilitate matching between modalities.

    Parameters:
    -----------
    barcodes : list or array
        List of cell barcodes
    strip_suffix : bool
        Whether to strip suffixes like '-1', '-2' that may be added to barcodes

    Returns:
    --------
    standardized : list
        List of standardized barcodes
    mapping : dict
        Mapping from standardized barcodes to original barcodes
    """
    standardized = []
    mapping = {}

    for barcode in barcodes:
        std_barcode = barcode

        # Strip common suffixes if requested
        if strip_suffix:
            # Match patterns like '-1', '-2', etc.
            if '-' in std_barcode:
                std_barcode = std_barcode.split('-')[0]

        standardized.append(std_barcode)
        mapping[std_barcode] = barcode

    return standardized, mapping


def standardize_barcodes(barcodes, strip_suffix=True):
    """
    Standardize cell barcodes to facilitate matching between modalities.

    Parameters:
    -----------
    barcodes : list or array
        List of cell barcodes
    strip_suffix : bool
        Whether to strip suffixes like '-1', '-2' that may be added to barcodes

    Returns:
    --------
    standardized : list
        List of standardized barcodes
    mapping : dict
        Mapping from standardized barcodes to original barcodes
    """
    standardized = []
    mapping = {}

    for barcode in barcodes:
        std_barcode = barcode

        # Strip common suffixes if requested
        if strip_suffix:
            # Match patterns like '-1', '-2', etc.
            if '-' in std_barcode:
                std_barcode = std_barcode.split('-')[0]

        standardized.append(std_barcode)
        mapping[std_barcode] = barcode

    return standardized, mapping


def match_cells(rna_path, atac_path, output_dir, match_strategy='intersection', force=False):
    """
    Match cells between RNA and ATAC data and save matched versions.

    Parameters:
    -----------
    rna_path : str
        Path to RNA h5ad file
    atac_path : str
        Path to ATAC h5ad file
    output_dir : str
        Directory to save matched files
    match_strategy : str
        Strategy for matching cells: 'intersection' (keep only common cells) or
        'union' (keep all cells, filling missing data with zeros)
    force : bool
        Whether to force matching even if cell counts are very different

    Returns:
    --------
    rna_matched_path : str
        Path to matched RNA file
    atac_matched_path : str
        Path to matched ATAC file
    """
    print(f"Loading RNA data from {rna_path}...")
    rna_adata = ad.read_h5ad(rna_path)

    print(f"Loading ATAC data from {atac_path}...")
    atac_adata = ad.read_h5ad(atac_path)

    print(f"RNA dataset has {rna_adata.n_obs} cells")
    print(f"ATAC dataset has {atac_adata.n_obs} cells")

    # Check if matching is necessary
    if rna_adata.n_obs == atac_adata.n_obs and np.all(rna_adata.obs_names == atac_adata.obs_names):
        print("Datasets already have matching cells. No need to match.")
        return rna_path, atac_path

    # Standardize barcodes to facilitate matching
    rna_std_barcodes, rna_mapping = standardize_barcodes(rna_adata.obs_names)
    atac_std_barcodes, atac_mapping = standardize_barcodes(atac_adata.obs_names)

    # Find common and unique barcodes
    common_barcodes = set(rna_std_barcodes).intersection(set(atac_std_barcodes))
    rna_only_barcodes = set(rna_std_barcodes) - common_barcodes
    atac_only_barcodes = set(atac_std_barcodes) - common_barcodes

    print(f"Found {len(common_barcodes)} cells common to both datasets")
    print(f"Found {len(rna_only_barcodes)} cells unique to RNA dataset")
    print(f"Found {len(atac_only_barcodes)} cells unique to ATAC dataset")

    # Safety check - if very few common cells, warn the user
    # if len(common_barcodes) < min(rna_adata.n_obs, atac_adata.n_obs) * 0.5 and not force:
    #     print("WARNING: Less than 50% of cells match between datasets.")
    #     print("This might indicate a problem with the data or barcode formats.")
    #     print("Consider checking the barcode formats or using force=True to proceed anyway.")
    #     return rna_path, atac_path

    if match_strategy == 'intersection':
        # Find indices of common cells in each dataset
        rna_common_indices = [i for i, bc in enumerate(rna_std_barcodes) if bc in common_barcodes]
        atac_common_indices = [i for i, bc in enumerate(atac_std_barcodes) if bc in common_barcodes]

        # Create new AnnData objects from the original data
        # This avoids issues with barcode formatting
        rna_matched = ad.AnnData(
            X=rna_adata.X[rna_common_indices],
            obs=rna_adata.obs.iloc[rna_common_indices].copy(),
            var=rna_adata.var.copy(),
            uns=rna_adata.uns.copy()
        )

        atac_matched = ad.AnnData(
            X=atac_adata.X[atac_common_indices],
            obs=atac_adata.obs.iloc[atac_common_indices].copy(),
            var=atac_adata.var.copy(),
            uns=atac_adata.uns.copy()
        )

        # Make sure the ordering is the same in both datasets
        # Map standardized barcodes to their indices in each dataset
        rna_bc_to_idx = {rna_std_barcodes[i]: i for i in rna_common_indices}
        atac_bc_to_idx = {atac_std_barcodes[i]: i for i in atac_common_indices}

        # Get a consistent ordering of common barcodes
        ordered_common_barcodes = sorted(common_barcodes)

        # Create ordered indices for both datasets
        ordered_rna_indices = [rna_bc_to_idx[bc] for bc in ordered_common_barcodes if bc in rna_bc_to_idx]
        ordered_atac_indices = [atac_bc_to_idx[bc] for bc in ordered_common_barcodes if bc in atac_bc_to_idx]

        # Create new AnnData objects with consistent cell ordering
        rna_matched = ad.AnnData(
            X=rna_adata.X[ordered_rna_indices],
            obs=rna_adata.obs.iloc[ordered_rna_indices].copy(),
            var=rna_adata.var.copy(),
            uns=rna_adata.uns.copy()
        )

        atac_matched = ad.AnnData(
            X=atac_adata.X[ordered_atac_indices],
            obs=atac_adata.obs.iloc[ordered_atac_indices].copy(),
            var=atac_adata.var.copy(),
            uns=atac_adata.uns.copy()
        )
    # Save matched datasets
    os.makedirs(output_dir, exist_ok=True)

    # Determine the sample type from the input filename
    rna_basename = os.path.basename(rna_path)
    sample_type = ""

    # Try to extract sample type (e.g., cll, healthy) from the filename
    if "cll" in rna_basename.lower():
        sample_type = "cll"
    elif "healthy" in rna_basename.lower():
        sample_type = "healthy"
    else:
        # If sample type can't be determined, use a generic prefix
        sample_type = "sample"

    # Use sample type in the matched filenames
    rna_matched_path = os.path.join(output_dir, f"{sample_type}_rna_matched.h5ad")
    atac_matched_path = os.path.join(output_dir, f"{sample_type}_atac_matched.h5ad")

    print(f"Saving matched RNA data ({rna_matched.n_obs} cells) to {rna_matched_path}...")
    rna_matched.write_h5ad(rna_matched_path)

    print(f"Saving matched ATAC data ({atac_matched.n_obs} cells) to {atac_matched_path}...")
    atac_matched.write_h5ad(atac_matched_path)

    return rna_matched_path, atac_matched_path

def process_data_directory(data_dir="./data", output_dir="./data/processed",
                           gene_pos_file=None, name2id_file=None,
                           match_cells_after=False):
    """
    Process all RNA-seq and ATAC-seq data files in the data directory.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the data files
    output_dir : str
        Path to the directory to save the processed files
    gene_pos_file : str, optional
        Path to gene positions CSV file containing chromosome and TSS info
    name2id_file : str, optional
        Path to gene name to Ensembl ID mapping CSV file
    match_cells_after : bool
        Whether to match cells between RNA and ATAC datasets after processing
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

    # To store paths of processed files for later cell matching
    processed_files = {}

    # Process each group
    for prefix, group_files in file_groups.items():
        print(f"Processing files with prefix {prefix}...")
        processed_files[prefix] = {}

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
            convert_rna_to_h5ad(rna_features, rna_barcodes, rna_matrix, rna_output,
                                gene_pos_file, name2id_file)
            processed_files[prefix]['rna'] = rna_output

        # Process ATAC-seq files if all are found
        if atac_barcodes and atac_matrix and atac_peaks:
            atac_output = os.path.join(output_dir, f"{prefix}_atac.h5ad")
            print(f"Converting ATAC-seq files to {atac_output}...")
            convert_atac_to_h5ad(atac_barcodes, atac_matrix, atac_peaks, atac_output)
            processed_files[prefix]['atac'] = atac_output

    # Match cells between RNA and ATAC datasets if requested
    if match_cells_after:
        matched_dir = os.path.join(output_dir, "matched")
        os.makedirs(matched_dir, exist_ok=True)

        for prefix, files in processed_files.items():
            if 'rna' in files and 'atac' in files:
                print(f"\nMatching cells for {prefix} dataset...")
                rna_matched, atac_matched = match_cells(
                    files['rna'],
                    files['atac'],
                    matched_dir,
                    match_strategy='intersection'
                )
                print(f"Matching completed. Matched files saved to {matched_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert RNA-seq or ATAC-seq data to h5ad format')
    parser.add_argument('--data_dir', default='./data', help='Directory containing the data files')
    parser.add_argument('--output_dir', default='./data/processed', help='Directory to save the processed files')
    parser.add_argument('--gene_pos_file', default=None,
                        help='Path to gene positions CSV file (all_gene_chrom_positions.csv)')
    parser.add_argument('--name2id_file', default=None,
                        help='Path to gene name to Ensembl ID mapping CSV file (name2id.csv)')
    parser.add_argument('--match_cells', action='store_true', help='Match cells between RNA and ATAC datasets')

    # Individual file processing
    parser.add_argument('--mode', choices=['rna', 'atac', 'auto'], default='auto',
                        help='Data type: rna for RNA-seq, atac for ATAC-seq, auto for automatic detection')
    parser.add_argument('--features', help='Path to features/genes file (.tsv.gz) - RNA-seq only')
    parser.add_argument('--peaks', help='Path to peaks file (.bed.gz) - ATAC-seq only')
    parser.add_argument('--barcodes', help='Path to barcodes file (.tsv.gz)')
    parser.add_argument('--matrix', help='Path to matrix file (.mtx.gz)')
    parser.add_argument('--output', help='Path to save output h5ad file')

    # Cell matching options
    parser.add_argument('--match_rna', help='Path to RNA h5ad file for cell matching')
    parser.add_argument('--match_atac', help='Path to ATAC h5ad file for cell matching')
    parser.add_argument('--match_output', default='./data/processed/matched', help='Directory to save matched files')
    parser.add_argument('--match_strategy', choices=['intersection', 'union'], default='intersection',
                        help='Strategy for matching cells')
    parser.add_argument('--force', action='store_true', help='Force matching even if cell counts are very different')

    args = parser.parse_args()

    # Cell matching mode
    if args.match_rna and args.match_atac:
        print(f"Matching cells between {args.match_rna} and {args.match_atac}...")
        match_cells(args.match_rna, args.match_atac, args.match_output,
                    args.match_strategy, args.force)

    # Individual file processing mode
    elif args.barcodes and args.matrix and (args.features or args.peaks):
        if args.mode == 'rna' or (args.mode == 'auto' and args.features):
            if not args.output:
                args.output = os.path.join(args.output_dir, 'rna.h5ad')
            convert_rna_to_h5ad(args.features, args.barcodes, args.matrix, args.output,
                                args.gene_pos_file, args.name2id_file)
        elif args.mode == 'atac' or (args.mode == 'auto' and args.peaks):
            if not args.output:
                args.output = os.path.join(args.output_dir, 'atac.h5ad')
            convert_atac_to_h5ad(args.barcodes, args.matrix, args.peaks, args.output)

    # Directory processing mode
    else:
        # Process all files in the data directory
        process_data_directory(args.data_dir, args.output_dir,
                               args.gene_pos_file, args.name2id_file,
                               args.match_cells)
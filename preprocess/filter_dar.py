import os
import argparse
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def find_intersection_peaks(cll_atac_path, healthy_atac_path, output_dir):
    """
    Find the intersection of peaks between CLL and healthy ATAC-seq datasets,
    and create new h5ad files with only these common peaks.

    Parameters:
    -----------
    cll_atac_path : str
        Path to the CLL ATAC-seq h5ad file
    healthy_atac_path : str
        Path to the healthy ATAC-seq h5ad file
    output_dir : str
        Directory to save the output h5ad files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the ATAC-seq datasets
    print(f"Loading CLL ATAC-seq data from {cll_atac_path}")
    cll_atac = ad.read_h5ad(cll_atac_path)
    print(f"CLL ATAC shape: {cll_atac.shape} (cells × peaks)")

    print(f"Loading healthy ATAC-seq data from {healthy_atac_path}")
    healthy_atac = ad.read_h5ad(healthy_atac_path)
    print(f"Healthy ATAC shape: {healthy_atac.shape} (cells × peaks)")

    # Check if observation names are unique
    print(f"CLL observations unique: {cll_atac.obs_names.is_unique}")
    print(f"Healthy observations unique: {healthy_atac.obs_names.is_unique}")

    # Make observation names unique if needed
    if not cll_atac.obs_names.is_unique:
        print("Making CLL observation names unique...")
        cll_atac.obs_names_make_unique()

    if not healthy_atac.obs_names.is_unique:
        print("Making healthy observation names unique...")
        healthy_atac.obs_names_make_unique()

    # Extract peak names
    cll_peaks = set(cll_atac.var_names)
    healthy_peaks = set(healthy_atac.var_names)

    # Find intersection
    common_peaks = list(cll_peaks.intersection(healthy_peaks))
    common_peaks.sort()  # Sort for reproducibility

    print(
        f"Found {len(common_peaks)} common peaks out of {len(cll_peaks)} CLL peaks and {len(healthy_peaks)} healthy peaks")

    # Create new AnnData objects with only common peaks
    cll_atac_common = cll_atac[:, common_peaks].copy()
    healthy_atac_common = healthy_atac[:, common_peaks].copy()

    print(f"CLL ATAC with common peaks shape: {cll_atac_common.shape} (cells × peaks)")
    print(f"Healthy ATAC with common peaks shape: {healthy_atac_common.shape} (cells × peaks)")

    # Get output file names
    cll_base = os.path.basename(cll_atac_path)
    healthy_base = os.path.basename(healthy_atac_path)

    cll_output = os.path.join(output_dir, cll_base.replace('.h5ad', '_common_peaks.h5ad'))
    healthy_output = os.path.join(output_dir, healthy_base.replace('.h5ad', '_common_peaks.h5ad'))

    # Save files
    print(f"Saving CLL data with common peaks to {cll_output}")
    cll_atac_common.write_h5ad(cll_output)

    print(f"Saving healthy data with common peaks to {healthy_output}")
    healthy_atac_common.write_h5ad(healthy_output)

    # Also save the list of common peaks
    peaks_file = os.path.join(output_dir, "common_peaks.txt")
    with open(peaks_file, 'w') as f:
        for peak in common_peaks:
            f.write(f"{peak}\n")
    print(f"Saved list of common peaks to {peaks_file}")

    # Create a combined dataset with all cells
    print("Creating a combined dataset with all cells and common peaks")

    # Check if the var DataFrames have the same columns
    cll_cols = set(cll_atac_common.var.columns)
    healthy_cols = set(healthy_atac_common.var.columns)
    common_cols = list(cll_cols.intersection(healthy_cols))

    # Add a sample_type column to obs
    cll_atac_common.obs['sample_type'] = 'CLL'
    healthy_atac_common.obs['sample_type'] = 'Healthy'

    # Concatenate the datasets
    combined = ad.concat(
        [cll_atac_common, healthy_atac_common],
        join='outer',  # Use outer join to keep all cells
        merge='same',  # Expect variables to match
        label='sample_type',  # Add source info to a new column
        keys=['CLL', 'Healthy']  # Keys for the samples
    )

    # Make sure the var index is properly set
    combined.var_names = common_peaks
    combined.var.index = common_peaks

    print(f"Combined dataset shape: {combined.shape} (cells × peaks)")

    # Save the combined dataset
    combined_output = os.path.join(output_dir, "combined_atac_common_peaks.h5ad")
    print(f"Saving combined dataset to {combined_output}")
    combined.write_h5ad(combined_output)

    return cll_atac_common, healthy_atac_common, combined


def main():
    parser = argparse.ArgumentParser(description='Find intersection of peaks between CLL and healthy ATAC-seq datasets')
    parser.add_argument('--cll', required=True, help='Path to CLL ATAC-seq h5ad file')
    parser.add_argument('--healthy', required=True, help='Path to healthy ATAC-seq h5ad file')
    parser.add_argument('--output_dir', default='./data/processed', help='Directory to save output h5ad files')

    args = parser.parse_args()

    find_intersection_peaks(args.cll, args.healthy, args.output_dir)


if __name__ == "__main__":
    main()
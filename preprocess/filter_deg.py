import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def filter_and_find_degs(rna_files_dir, output_dir='./results/degs', min_cells=10, min_genes=200,
                         n_top_genes=2000, pval_cutoff=0.05, logfc_cutoff=0.5):
    """
    Load RNA h5ad files, filter cells and genes, normalize, find DEGs, and save results.
    Also saves a version of the data with only DEGs without scaling or normalization.

    Parameters:
    -----------
    rna_files_dir: str
        Directory containing RNA h5ad files
    output_dir: str
        Directory to save DEG results and plots
    min_cells: int
        Minimum number of cells expressing a gene for it to be kept
    min_genes: int
        Minimum number of genes detected in a cell for it to be kept
    n_top_genes: int
        Number of highly variable genes to select
    pval_cutoff: float
        p-value threshold for DEG significance
    logfc_cutoff: float
        Log fold change threshold for DEG significance
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all h5ad files in the directory
    h5ad_files = [f for f in os.listdir(rna_files_dir) if f.endswith('_rna.h5ad')]

    if not h5ad_files:
        print(f"No RNA h5ad files found in {rna_files_dir}")
        return

    # Separate files by sample type
    healthy_files = [f for f in h5ad_files if 'healthy' in f]
    cll_files = [f for f in h5ad_files if 'cll' in f]

    if not healthy_files:
        print("No healthy samples found")
        return

    if not cll_files:
        print("No CLL samples found")
        return

    print(f"Found {len(healthy_files)} healthy and {len(cll_files)} CLL samples")

    # Load and merge data
    adatas = []
    original_adatas = {}  # Store the original data before processing

    # Load healthy samples
    for h_file in healthy_files:
        file_path = os.path.join(rna_files_dir, h_file)
        print(f"Loading healthy sample: {h_file}")
        adata = sc.read_h5ad(file_path)
        print(adata)
        original_adatas[h_file] = adata.copy()  # Store original
        adata.obs['sample_type'] = 'healthy'
        adata.obs['sample_id'] = h_file.split('_')[0]  # Extract GSM ID
        adatas.append(adata)

    # Load CLL samples
    for c_file in cll_files:
        file_path = os.path.join(rna_files_dir, c_file)
        print(f"Loading CLL sample: {c_file}")
        adata = sc.read_h5ad(file_path)
        original_adatas[c_file] = adata.copy()  # Store original
        adata.obs['sample_type'] = 'cll'
        adata.obs['sample_id'] = c_file.split('_')[0]  # Extract GSM ID
        adatas.append(adata)

    # Concatenate all samples
    if len(adatas) > 1:
        adata = adatas[0].concatenate(adatas[1:], join='outer')
    else:
        adata = adatas[0]

    print(f"Combined dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Basic QC and filtering
    print("Performing QC and filtering...")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Plot QC metrics
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    sns.histplot(adata.obs['n_genes_by_counts'], kde=False, ax=axs[0, 0])
    axs[0, 0].set_title('Genes per Cell')
    axs[0, 0].axvline(min_genes, color='red')

    sns.histplot(adata.obs['total_counts'], kde=False, ax=axs[0, 1])
    axs[0, 1].set_title('UMI Counts per Cell')

    sns.histplot(adata.obs['pct_counts_mt'] if 'pct_counts_mt' in adata.obs else [0] * adata.shape[0],
                 kde=False, ax=axs[0, 2])
    axs[0, 2].set_title('Percent Mitochondrial')

    sc.pl.violin(adata, 'n_genes_by_counts', groupby='sample_type', ax=axs[1, 0], show=False)
    sc.pl.violin(adata, 'total_counts', groupby='sample_type', ax=axs[1, 1], show=False)
    sc.pl.violin(adata, 'pct_counts_mt' if 'pct_counts_mt' in adata.obs else 'n_genes_by_counts',
                 groupby='sample_type', ax=axs[1, 2], show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qc_metrics.png'))
    plt.close()

    # Filter cells and genes
    print(f"Before filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Filter mitochondrial genes if they're annotated
    if 'mt' in adata.var_names[0]:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < 20, :]

    print(f"After filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Create a copy of the filtered but unnormalized data
    unnormalized_adata = adata.copy()

    # Normalize data for DEG analysis
    print("Normalizing data for DEG analysis...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    print(f"Identifying top {n_top_genes} variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Plot highly variable genes
    plt.figure(figsize=(10, 7))
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(output_dir, 'highly_variable_genes.png'))
    plt.close()

    # Find DEGs using Wilcoxon rank-sum test
    print("Computing differential expression between healthy and CLL...")
    sc.tl.rank_genes_groups(adata, groupby='sample_type', method='wilcoxon')

    # Get results and save to CSV
    print("Saving DEG results...")
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    # Create a DataFrame with all DEG results
    deg_data = []
    for group in groups:
        genes = result['names'][group]
        scores = result['scores'][group]
        pvals = result['pvals'][group]
        pvals_adj = result['pvals_adj'][group]
        logfcs = result['logfoldchanges'][group]

        for i in range(len(genes)):
            deg_data.append({
                'group': group,
                'gene': genes[i],
                'score': scores[i],
                'pval': pvals[i],
                'pval_adj': pvals_adj[i],
                'logfc': logfcs[i]
            })

    deg_df = pd.DataFrame(deg_data)

    # Save full results
    deg_df.to_csv(os.path.join(output_dir, 'all_degs.csv'), index=False)

    # Filter for significant DEGs
    sig_deg_df = deg_df[(deg_df['pval_adj'] < pval_cutoff) & (abs(deg_df['logfc']) > logfc_cutoff)]
    sig_deg_df.to_csv(os.path.join(output_dir, 'significant_degs.csv'), index=False)

    # Plot top DEGs
    plt.figure(figsize=(12, 8))
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
    plt.savefig(os.path.join(output_dir, 'top_degs.png'))
    plt.close()

    # Create volcano plot for each comparison
    for group in groups:
        group_degs = deg_df[deg_df['group'] == group]

        plt.figure(figsize=(10, 8))
        plt.scatter(
            group_degs['logfc'],
            -np.log10(group_degs['pval']),
            alpha=0.5,
            s=5
        )

        # Highlight significant genes
        sig_degs = group_degs[(group_degs['pval_adj'] < pval_cutoff) & (abs(group_degs['logfc']) > logfc_cutoff)]
        plt.scatter(
            sig_degs['logfc'],
            -np.log10(sig_degs['pval']),
            alpha=0.7,
            s=10,
            color='red'
        )

        # Label top genes
        top_genes = sig_degs.sort_values('pval').head(20)
        for _, row in top_genes.iterrows():
            plt.annotate(
                row['gene'],
                (row['logfc'], -np.log10(row['pval'])),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points'
            )

        plt.axhline(-np.log10(pval_cutoff), linestyle='--', color='gray')
        plt.axvline(logfc_cutoff, linestyle='--', color='gray')
        plt.axvline(-logfc_cutoff, linestyle='--', color='gray')

        plt.xlabel('Log2 fold change')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Volcano plot - {group}')

        plt.savefig(os.path.join(output_dir, f'volcano_plot_{group}.png'))
        plt.close()

    # Heatmap of top DEGs
    n_top = 25
    top_degs = {}
    for group in groups:
        group_degs = deg_df[deg_df['group'] == group]
        top_degs[group] = group_degs.sort_values('pval').head(n_top)['gene'].tolist()

    # Combine unique top DEGs from all groups
    all_top_degs = []
    for group in groups:
        all_top_degs.extend(top_degs[group])
    all_top_degs = list(set(all_top_degs))

    # Plot heatmap for top DEGs
    if len(all_top_degs) > 0:
        plt.figure(figsize=(12, len(all_top_degs) * 0.25 + 2))
        sc.pl.heatmap(adata, all_top_degs, groupby='sample_type', show=False)
        plt.savefig(os.path.join(output_dir, 'top_degs_heatmap.png'))
        plt.close()

    print(f"Identified {len(sig_deg_df)} significant DEGs (p-adj < {pval_cutoff}, |logFC| > {logfc_cutoff})")

    # Save AnnData objects with only significant DEGs (unnormalized)
    significant_genes = set(sig_deg_df['gene'].unique())
    print(f"Saving AnnData objects with only significant DEGs (unnormalized)...")

    # For each original file, save a version with only DEGs
    for file_name, orig_adata in original_adatas.items():
        # Filter to keep only significant DEGs
        mask = [gene in significant_genes for gene in orig_adata.var_names]
        if sum(mask) > 0:  # Only save if there are DEGs present
            # Subset the original unnormalized data to keep only DEGs
            deg_adata = orig_adata[:, mask].copy()

            # Name the output file
            output_file = os.path.join(output_dir, file_name.replace('.h5ad', '_degs.h5ad'))

            # Save the DEG-only data (unnormalized)
            deg_adata.write_h5ad(output_file)
            print(f"Saved {sum(mask)} DEGs for {file_name} to {output_file}")

    print(f"Results saved to {output_dir}")

    return adata, sig_deg_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Find differentially expressed genes between healthy and CLL samples')
    parser.add_argument('--data_dir', default='../data/processed',
                        help='Directory containing processed h5ad files')
    parser.add_argument('--output_dir', default='../data/processed',
                        help='Directory to save DEG results')
    parser.add_argument('--min_cells', type=int, default=3,
                        help='Minimum number of cells for a gene to be kept')
    parser.add_argument('--min_genes', type=int, default=200,
                        help='Minimum number of genes for a cell to be kept')
    parser.add_argument('--n_top_genes', type=int, default=3000,
                        help='Number of highly variable genes to select')
    parser.add_argument('--pval_cutoff', type=float, default=0.15,
                        help='P-value cutoff for DEG significance')
    parser.add_argument('--logfc_cutoff', type=float, default=0.2,
                        help='Log fold change cutoff for DEG significance')

    args = parser.parse_args()

    filter_and_find_degs(
        args.data_dir,
        args.output_dir,
        args.min_cells,
        args.min_genes,
        args.n_top_genes,
        args.pval_cutoff,
        args.logfc_cutoff
    )
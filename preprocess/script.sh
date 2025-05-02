

python differential_grn_preprocess.py \
  --mode atac \
  --barcodes ../data/GSM4829412_cll_atac_filtered_barcodes.tsv.gz \
  --matrix ../data/GSM4829412_cll_atac_filtered_matrix.mtx.gz \
  --peaks ../data/GSM4829412_cll_atac_peaks.bed.gz \
  --output ../data/processed/cll_atac.h5ad

  python differential_grn_preprocess.py \
  --mode rna \
  --features ../data/GSM4829413_cll_rna_features.tsv.gz \
  --barcodes ../data/GSM4829413_cll_rna_filtered_barcodes.tsv.gz \
  --matrix ../data/GSM4829413_cll_rna_filtered_matrix.mtx.gz \
  --output ../data/processed/cll_rna.h5ad

python differential_grn_preprocess.py \
  --mode atac \
  --barcodes ../data/GSM4829410_healthy_atac_filtered_barcodes.tsv.gz \
  --matrix ../data/GSM4829410_healthy_atac_filtered_matrix.mtx.gz \
  --peaks ../data/GSM4829410_healthy_atac_peaks.bed.gz \
  --output ../data/processed/healthy_atac.h5ad

  python differential_grn_preprocess.py \
  --mode rna \
  --features ../data/GSM4829411_healthy_rna_features.tsv.gz \
  --barcodes ../data/GSM4829411_healthy_rna_filtered_barcodes.tsv.gz \
  --matrix ../data/GSM4829411_healthy_rna_filtered_matrix.mtx.gz \
  --output ../data/processed/healthy_rna.h5ad


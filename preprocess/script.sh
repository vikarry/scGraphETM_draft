

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
  --output ../data/processed/cll_rna.h5ad \
  --gene_pos_file ../data/all_gene_chrom_positions.csv \
  --name2id_file ../data/name2id.csv

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
  --output ../data/processed/healthy_rna.h5ad \
  --gene_pos_file ../data/all_gene_chrom_positions.csv \
  --name2id_file ../data/name2id.csv

python process_tg_re.py \
  --rna_path  ../data/processed/cll_rna_degs.h5ad \
  --atac_path  ../data/processed/cll_atac.h5ad \
  --flag nearby \
  --top 10 \
  --distance 250000 \
  --output_dir  ../data/processed/

python process_cistarget_tf_re.py \
  --rna_path ../data/processed/cll_rna_degs.h5ad \
  --atac_path ../data/processed/cll_tg_re_nearby_dist_1m_ATAC.h5ad \
  --tg_re_matrix ../data/processed/cll_tg_re_nearby_dist_250000bp_matrix.pkl \
  --cistarget_score ../../data/hg38_screen_v10_clust.regions_vs_motifs.scores.feather \
  --motif2tf ../../data/to/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
  --threshold 3 \
  --output_dir ../data/processed/

python process_tg_re.py \
--rna_path  ../data/processed/healthy_rna_degs.h5ad \
--atac_path  ../data/processed/healthy_atac.h5ad \
--flag nearby \
--top 10 \
--distance 2500000 \
--output_dir  ../data/processed/
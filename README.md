# scGraphETM


## Generating H5AD Files
To properly convert raw single cell RNA or ATAC data into H5ad format which is required to run scGraphETM use [here](./build_grn/convert_raw_toh5ad.py).

## Build TG-RE Connections [here](./build_grn/process_tg_re.py)
### Variables
- **`dataset`**: Label for the dataset used in the analysis
- **`flag`**: Method flag to select the analysis method
  - **'`nearby `**: Add an edge between a TG and RE if the peak is withing a distance threshold from the TSS of the gene
  - **'`gbm `**: Use gradient boosting to select top number of most relevant peaks by predicting gene expression from peak expression 
  - **'`both `**: Use both methods to connect TG and RE
- **`top`**: Number of top peaks to select
- **`distance`**: Threshold distance within TSS of a gene

### Inputs
- **RNA.h5ad**
- **ATAC.h5ad**

### Outputs
- **Filtered ATAC-seq Dataset**: Subset of the ATAC-seq data corresponding to the matrix generated
- **TG-RE Connectivity Matrix**: A sparse matrix indicating gene-peak connections, saved as a pickle file.

## Build TF-RE Connections [here](./build_grn/process_cistarget_re.py)
Calculates the overlap percentage between available CisTarget region and peaks. Identifies and matches corresponding peaks to TFs based on a threshold score value.

### Inputs
- **RNA.h5ad**
- **ATAC.h5ad**
- **Score Data**: Path to the CisTarget feather format file containing scores indicating the significance of motif-regions.
- **Motif to TF Mapping**: CSV file mapping motifs to transcription factors

## Output
- **TF-RE Connectivity Matrices**: Sparse matrices representing significant interactions between genes and peaks.
- **GRN Matrix**: A merged representation of TG-RE and TF-RE matrices for the overall gene regulatory network.

## Project Structure
```
├── data/
│   ├── processed/
│   │   ├── cll_atac.h5ad
│   │   ├── cll_rna.h5ad
│   │   ├── healthy_atac.h5ad
│   │   └── healthy_rna.h5ad
│   ├── GSM4829410_healthy_atac_filtered_barcodes.tsv.gz
│   ├── GSM4829410_healthy_atac_filtered_matrix.mtx.gz
│   ├── GSM4829410_healthy_atac_peaks.bed.gz
│   ├── GSM4829411_healthy_rna_features.tsv.gz
│   ├── GSM4829411_healthy_rna_filtered_barcodes.tsv.gz
│   ├── GSM4829411_healthy_rna_filtered_matrix.mtx.gz
│   ├── GSM4829412_cll_atac_filtered_barcodes.tsv.gz
│   ├── GSM4829412_cll_atac_filtered_matrix.mtx.gz
│   ├── GSM4829412_cll_atac_peaks.bed.gz
│   ├── GSM4829413_cll_rna_features.tsv.gz
│   ├── GSM4829413_cll_rna_filtered_barcodes.tsv.gz
│   └── GSM4829413_cll_rna_filtered_matrix.mtx.gz
├── preprocess/
│   ├── convert_raw_to_h5ad.py      # Convert raw files to h5ad format
│   ├── differential_grn_preprocess.py  # Preprocess for differential GRN analysis
│   ├── filter_deg.py               # Filter differentially expressed genes
│   ├── process_cistarget_tf_re.py  # Process TF-RE connections using CisTarget
│   └── process_tg_re.py            # Process TG-RE connections
├── evaluation.py                    # Model evaluation functions
├── imputation.py                    # Imputation of missing data
├── main.py                          # Main script to run the model
├── model.py                         # Model definition
├── modules.py                       # Neural network modules
├── scDataloader.py                  # Data loader for single-cell data
├── script.sh                        # Example shell script for running pipeline
├── tester.py                        # Testing functions
├── train.py                         # Training functions
└── train_node2vec.py                # Script to train node2vec embeddings
```
# scGraphETM


## Generating H5AD Files
To properly convert raw single cell RNA or ATAC data into H5ad format which is required to run scGraphETM use [here](./build_grn/convert_raw_toh5ad.py).

## Build Target Gene - Region Connections [here](./build_grn/process_tg_re.py)
### Variables
- **`dataset`**: Label for the dataset used in the analysis
- **`flag`**: Method flag to select the analysis method
  - **'`nearby `**: Add an edge between a TG and RE if the peak is withing a distance threshold from the TSS of the gene
  - **'`gbm `**: Use gradient boosting to select top number of most relevant peaks by predicting gene expression from peak expression 
  - **'`both `**: Use both methods to connect TG and RE
- **`top`**: Number of top peaks to select
- **`distance`**: Threshold distance within TSS of a gene

## Outputs
- **Filtered ATAC-seq Dataset**: Subset of the ATAC-seq data corresponding to the matrix generated
- **Gene-Peak Connectivity Matrix**: A sparse matrix indicating gene-peak connections, saved as a pickle file.


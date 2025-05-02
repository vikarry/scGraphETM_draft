import random
import numpy as np
import torch
import anndata as ad
import scanpy as sc
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScDataset(Dataset):
    def __init__(
            self,
            rna_adata,
            atac_adata,
            num_of_gene,
            num_of_peak,
            rna_adata_gnn,
            atac_adata_gnn,
            emb_size=512
    ):
        self.indices = range(len(rna_adata))
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        self.rna_adata_gnn = rna_adata_gnn
        self.atac_adata_gnn = atac_adata_gnn

        self.num_of_gene = num_of_gene
        self.num_of_peak = num_of_peak

        # Process data
        self.rna_adata, self.X_rna_tensor, self.X_rna_tensor_normalized = (
            self.process_mod(self.rna_adata, self.num_of_gene, False))
        self.atac_adata, self.X_atac_tensor, self.X_atac_tensor_normalized = (
            self.process_mod(self.atac_adata, self.num_of_peak, False))

        self.rna_adata_gnn, self.X_rna_tensor_gnn, self.X_rna_tensor_normalized_gnn = (
            self.process_mod(self.rna_adata_gnn, None, True)
        )
        self.atac_adata_gnn, self.X_atac_tensor_gnn, self.X_atac_tensor_normalized_gnn = (
            self.process_mod(self.atac_adata_gnn, None, True)
        )

    @staticmethod
    def process_mod(adata, num_of_mod, is_gnn_adata):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        X = adata.X.toarray()

        X_tensor = torch.from_numpy(X).float()
        sums = X_tensor.sum(1).unsqueeze(1)
        X_tensor_normalized = X_tensor / sums

        return adata, X_tensor, X_tensor_normalized

    @staticmethod
    def custom_collate_fn(batch):
        (rna_batch, rna_batch_normalized,
         atac_batch, atac_batch_normalized,
         rna_batch_gnn, rna_batch_normalized_gnn,
         atac_batch_gnn, atac_batch_normalized_gnn, indices) = zip(*batch)

        rna_batch = torch.stack(rna_batch)
        rna_batch_normalized = torch.stack(rna_batch_normalized)
        atac_batch = torch.stack(atac_batch)
        atac_batch_normalized = torch.stack(atac_batch_normalized)

        rna_batch_gnn = torch.stack(rna_batch_gnn)
        rna_batch_normalized_gnn = torch.stack(rna_batch_normalized_gnn)
        atac_batch_gnn = torch.stack(atac_batch_gnn)
        atac_batch_normalized_gnn = torch.stack(atac_batch_normalized_gnn)

        return (rna_batch, rna_batch_normalized,
                atac_batch, atac_batch_normalized,
                rna_batch_gnn, rna_batch_normalized_gnn,
                atac_batch_gnn, atac_batch_normalized_gnn)

    def __len__(self):
        return self.rna_adata.shape[0]

    def __getitem__(self, idx):
        rna_item = self.X_rna_tensor[idx]
        rna_item_normalized = self.X_rna_tensor_normalized[idx]
        atac_item = self.X_atac_tensor[idx]
        atac_item_normalized = self.X_atac_tensor_normalized[idx]
        rna_item_gnn = self.X_rna_tensor_gnn[idx]
        rna_item_normalized_gnn = self.X_rna_tensor_normalized_gnn[idx]
        atac_item_gnn = self.X_atac_tensor_gnn[idx]
        atac_item_normalized_gnn = self.X_atac_tensor_normalized_gnn[idx]

        return (rna_item, rna_item_normalized,
                atac_item, atac_item_normalized,
                rna_item_gnn, rna_item_normalized_gnn,
                atac_item_gnn, atac_item_normalized_gnn, self.indices[idx])

    def get_all_data(self, indices):
        return {
            'X_rna_tensor_normalized': self.X_rna_tensor_normalized[indices],
            'X_atac_tensor_normalized': self.X_atac_tensor_normalized[indices],
            'X_rna_tensor_unnormalized': self.X_rna_tensor[indices],
            'X_atac_tensor_unnormalized': self.X_atac_tensor[indices],
            'rna_adata': self.rna_adata[indices],
            'atac_adata': self.atac_adata[indices],
            'indices': indices
        }


class ScDataLoader:
    def __init__(
            self,
            data_dir,
            batch_size=32,
            seed=42,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            train_ratio=0.8
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Set seed for reproducibility
        set_seed(seed)

        # Load RNA and ATAC data
        self.rna_adata = self.load_rna_data()
        self.atac_adata = self.load_atac_data()
        self.rna_adata_gnn = self.load_rna_gnn_data()
        self.atac_adata_gnn = self.load_atac_gnn_data()

        # Get dimensions
        self.num_of_gene = self.rna_adata.shape[1]
        self.num_of_peak = self.atac_adata.shape[1]
        self.num_of_gene_gnn = self.rna_adata_gnn.shape[1]
        self.num_of_peak_gnn = self.atac_adata_gnn.shape[1]

        # Create dataset
        self.scdataset = ScDataset(
            self.rna_adata,
            self.atac_adata,
            self.num_of_gene,
            self.num_of_peak,
            self.rna_adata_gnn,
            self.atac_adata_gnn,
            emb_size=128
        )

        # Split into train and test sets
        train_size = int(train_ratio * len(self.scdataset))
        test_size = len(self.scdataset) - train_size
        self.train_dataset, self.test_dataset = random_split(self.scdataset, [train_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.scdataset.custom_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.scdataset.custom_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    def load_rna_data(self):
        """Load RNA data from data directory"""
        rna_path = f"{self.data_dir}/rna.h5ad"
        return ad.read_h5ad(rna_path)

    def load_atac_data(self):
        """Load ATAC data from data directory"""
        atac_path = f"{self.data_dir}/atac.h5ad"
        return ad.read_h5ad(atac_path)

    def load_rna_gnn_data(self):
        """Load RNA GNN data from data directory"""
        rna_gnn_path = f"{self.data_dir}/rna_gnn.h5ad"
        return ad.read_h5ad(rna_gnn_path)

    def load_atac_gnn_data(self):
        """Load ATAC GNN data from data directory"""
        atac_gnn_path = f"{self.data_dir}/atac_gnn.h5ad"
        return ad.read_h5ad(atac_gnn_path)

    def get_gene_num(self):
        """Get number of genes"""
        return self.num_of_gene

    def get_peak_num(self):
        """Get number of peaks"""
        return self.num_of_peak

    def get_gene_num_gnn(self):
        """Get number of genes in GNN data"""
        return self.num_of_gene_gnn

    def get_peak_num_gnn(self):
        """Get number of peaks in GNN data"""
        return self.num_of_peak_gnn

    def get_rna_adata_gnn(self):
        """Get RNA GNN AnnData object"""
        return self.rna_adata_gnn

    def get_atac_adata_gnn(self):
        """Get ATAC GNN AnnData object"""
        return self.atac_adata_gnn

    def get_all_train_data(self):
        """Get all training data"""
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'indices'):
            train_indices = self.train_dataset.indices
        else:
            train_indices = range(len(self.scdataset))
        return self.scdataset.get_all_data(train_indices)

    def get_all_test_data(self):
        """Get all test data"""
        if hasattr(self, 'test_loader'):
            test_indices = self.test_dataset.indices
            return self.scdataset.get_all_data(test_indices)
        else:
            return None

    def train_dataloader(self):
        """Get training data loader"""
        return self.train_loader

    def test_dataloader(self):
        """Get test data loader"""
        return self.test_loader
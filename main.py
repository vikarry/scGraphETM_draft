import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

from sc_model import ScModel
from scDataLoader import ScDataLoader, set_seed
from train import train_model
from grn_evaluation import evaluate_grn
from imputation import impute


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ScModel training and evaluation')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the data')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory to save results')

    # Model parameters
    parser.add_argument('--emb_size', type=int, default=128,
                        help='Embedding size')
    parser.add_argument('--num_topics', type=int, default=50,
                        help='Number of topics')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--shared_decoder', action='store_true',
                        help='Use shared decoder')
    parser.add_argument('--use_gnn', action='store_true',
                        help='Use graph neural network')
    parser.add_argument('--use_xtrimo', action='store_true',
                        help='Use xTrimoGene embedding')
    parser.add_argument('--use_graph_recon', action='store_true',
                        help='Use graph reconstruction loss')
    parser.add_argument('--graph_recon_weight', type=float, default=1.0,
                        help='Weight for graph reconstruction loss')
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='Positive weight for graph reconstruction loss')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--best_model_path', type=str, default=None,
                        help='Path to best model to load')

    # Evaluation parameters
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_eval', 'eval_grn', 'eval_imputation', 'full'],
                        help='Mode of operation')
    parser.add_argument('--grn_matrix_path', type=str, default=None,
                        help='Path to GRN matrix for evaluation')
    parser.add_argument('--bed_file_path', type=str, default=None,
                        help='Path to bed file for GRN evaluation')

    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.result_dir, f"run_{timestamp}")

    directories = {
        'run_dir': run_dir,
        'model_dir': os.path.join(run_dir, 'models'),
        'plot_dir': os.path.join(run_dir, 'plots'),
        'umap_dir': os.path.join(run_dir, 'plots', 'umap'),
        'tsne_dir': os.path.join(run_dir, 'plots', 'tsne'),
        'grn_dir': os.path.join(run_dir, 'plots', 'grn'),
        'imputation_dir': os.path.join(run_dir, 'plots', 'imputation')
    }

    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)

    return directories


def save_config(args, directories):
    """Save configuration to a file."""
    config_path = os.path.join(directories['run_dir'], 'config.txt')

    with open(config_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    directories = setup_directories(args)
    save_config(args, directories)

    data_loader = ScDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )

    num_of_gene = data_loader.get_gene_num()
    num_of_peak = data_loader.get_peak_num()
    num_of_gene_gnn = data_loader.get_gene_num_gnn()
    num_of_peak_gnn = data_loader.get_peak_num_gnn()
    node2vec = data_loader.get_node2vec_embedding()

    print(f"Data loaded with {num_of_gene} genes and {num_of_peak} peaks")

    model = ScModel(
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        num_of_gene_gnn=num_of_gene_gnn,
        num_of_peak_gnn=num_of_peak_gnn,
        emb_size=args.emb_size,
        num_of_topic=args.num_topics,
        device=device,
        batch_size=args.batch_size,
        result_save_path=(directories['umap_dir'], directories['tsne_dir']),
        lr=args.lr,
        best_model_path=args.best_model_path,
        use_graph_recon=args.use_graph_recon,
        graph_recon_weight=args.graph_recon_weight,
        pos_weight=args.pos_weight,
        node2vec=node2vec,
        use_gnn=args.use_gnn,
        use_xtrimo=args.use_xtrimo,
        shared_decoder=args.shared_decoder
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Training mode
    if args.mode in ['train', 'train_eval', 'full']:
        print("Starting training...")
        model = train_model(model, data_loader, epochs=args.epochs)

        # Save best model
        best_model_path = os.path.join(directories['model_dir'], 'best_model.pt')
        model.save(best_model_path)
        print(f"Best model saved to {best_model_path}")

    # Load best model for evaluation if available
    if args.mode in ['train_eval', 'eval_grn', 'eval_imputation', 'full']:
        # First try to use the best model from training if available
        if model.best_model is not None:
            model.load_state_dict(model.best_model)
            print("Loaded best model from training for evaluation")
        # Otherwise, try to load from path
        elif args.best_model_path:
            model.load_best_model(args.best_model_path)
            print(f"Loaded best model from {args.best_model_path}")

    # GRN evaluation mode
    if args.mode in ['eval_grn', 'full']:
        print("Evaluating GRN...")

        if args.grn_matrix_path is None or args.bed_file_path is None:
            print("GRN evaluation requires grn_matrix_path and bed_file_path arguments")
        else:
            rna_adata_gnn = data_loader.get_rna_adata_gnn()
            atac_adata_gnn = data_loader.get_atac_adata_gnn()

            evaluate_grn(
                model,
                rna_adata_gnn,
                atac_adata_gnn,
                args.grn_matrix_path,
                args.bed_file_path,
                directories['grn_dir']
            )
            print("GRN evaluation completed")

    if args.mode in ['eval_imputation', 'full']:
        print("Evaluating imputation...")

        results = impute(model, data_loader, directories['imputation_dir'])

        for modality, scores in results.items():
            print(f"\n{modality} Imputation Results:")
            for metric, value in scores.items():
                print(f"  {metric.capitalize()}: {value:.4f}")

        print("Imputation evaluation completed")

    print(f"All operations completed. Results saved to {directories['run_dir']}")


if __name__ == "__main__":
    main()
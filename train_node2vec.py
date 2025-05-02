import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import add_self_loops
import networkx as nx


def load_adjacency_matrix(path):
    """
    Load adjacency matrix from specified path
    """
    print(f"Loading adjacency matrix from {path}")
    if path.endswith('.pkl'):
        return pd.read_pickle(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path, index_col=0)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def adjacency_to_edge_index(adj_matrix):
    """
    Convert adjacency matrix to edge index format for PyTorch Geometric
    """
    # If it's a DataFrame, convert to numpy array
    if isinstance(adj_matrix, pd.DataFrame):
        node_names = adj_matrix.index
        node_index = {node: idx for idx, node in enumerate(node_names)}
        adj_np = adj_matrix.values
    else:
        adj_np = adj_matrix
        node_index = {i: i for i in range(adj_np.shape[0])}

    if adj_np.shape[0] != adj_np.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj_np.shape}")

    # Get edges from adjacency matrix
    sources = []
    targets = []
    rows, cols = np.where(adj_np > 0)
    for i in range(len(rows)):
        sources.append(rows[i])
        targets.append(cols[i])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return edge_index, node_index


def is_disconnected(edge_index, num_nodes):
    """
    Check if the graph has disconnected components
    """
    edge_list = edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    num_components = nx.number_connected_components(G)

    if num_components > 1:
        component_sizes = [len(c) for c in nx.connected_components(G)]
        print(f"Graph has {num_components} connected components")
        print(f"Component sizes: {component_sizes}")
        return True

    return False


def calculate_sparsity(edge_index, num_nodes):
    """
    Calculate the sparsity of the graph
    """
    num_edges = edge_index.size(1)
    max_possible_edges = num_nodes * (num_nodes - 1) / 2  # For undirected graph
    sparsity = num_edges / max_possible_edges

    print(f"Graph sparsity: {sparsity:.6f} ({num_edges} edges out of {max_possible_edges:.0f} possible)")

    if sparsity < 0.01:
        print("WARNING: Very sparse graph detected (<1% of possible edges)")
    elif sparsity < 0.05:
        print("INFO: Sparse graph detected (<5% of possible edges)")

    return sparsity


def train_node2vec(edge_index, num_nodes, embedding_dim=128, walk_length=40,
                   context_size=8, walks_per_node=15, p=0.5, q=2.0,
                   num_epochs=250, batch_size=64, learning_rate=0.01,
                   use_gpu=True, patience=20):
    """
    Train a Node2Vec model optimized for sparse graphs
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Training on CPU")

    sparsity = calculate_sparsity(edge_index, num_nodes)

    is_discon = is_disconnected(edge_index, num_nodes)
    if is_discon:
        print("WARNING: Graph has disconnected components. Embeddings may not be meaningful across components.")

    # Add self-loops to improve random walk behavior for sparse graphs
    edge_index_with_loops = add_self_loops(edge_index)[0]
    print(f"Added self-loops. New edge count: {edge_index_with_loops.size(1)}")

    # Move edge_index to device
    edge_index_with_loops = edge_index_with_loops.to(device)

    # Initialize Node2Vec model
    model = Node2Vec(
        edge_index_with_loops,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        sparse=True
    ).to(device)

    # Create DataLoader with appropriate workers
    workers = min(4, os.cpu_count() or 1)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=workers)

    # Initialize optimizer
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, verbose=True
    )

    def train_epoch():
        model.train()
        total_loss = 0
        num_batches = 0

        for pos_rw, neg_rw in loader:
            if pos_rw.size(0) == 0 or neg_rw.size(0) == 0:
                continue

            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return float('inf')

        return total_loss / num_batches

    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        try:
            loss = train_epoch()
            scheduler.step(loss)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Check for improvement
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping check
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        except Exception as e:
            print(f"Error during training epoch {epoch}: {str(e)}")
            if best_model_state is not None:
                print("Restoring best model from previous epoch")
                model.load_state_dict(best_model_state)
            else:
                print("No valid model state to restore. Training failed.")
                raise e

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    model.eval()
    with torch.no_grad():
        embeddings = model()

    print(f"Final embeddings shape: {embeddings.size()}")
    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser(description='Train Node2Vec embeddings for graphs (optimized for sparse graphs)')
    parser.add_argument('--modality', type=str, required=True, choices=['cond', 'icd'],
                        help='Modality selector (cond or icd)')
    parser.add_argument('--output_dir', type=str, default='./data/',
                        help='Directory to save outputs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of the embeddings')
    parser.add_argument('--walk_length', type=int, default=40,
                        help='Length of each random walk')
    parser.add_argument('--context_size', type=int, default=8,
                        help='Context window size')
    parser.add_argument('--walks_per_node', type=int, default=15,
                        help='Number of walks per node')
    parser.add_argument('--p', type=float, default=0.5,
                        help='Return parameter (controls likelihood of revisiting nodes)')
    parser.add_argument('--q', type=float, default=2.0,
                        help='In-out parameter (controls exploration vs exploitation)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs with no improvement)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training even if GPU is available')
    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Direct path to adjacency matrix (overrides modality-based path)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.matrix_path:
        matrix_path = args.matrix_path
    else:
        matrix_path = f'./data/{args.modality}_adj_matrix.pkl'

    output_base = os.path.join(args.output_dir, f'{args.modality}_sparse_node_embeddings_{args.embedding_dim}')
    mapping_output = os.path.join(args.output_dir, f'{args.modality}_node_mapping_{args.embedding_dim}.csv')

    output_pt = f"{output_base}.pt"
    output_named = f"{output_base}_named.pkl"
    output_np = f"{output_base}.npy"

    # Load adjacency matrix
    try:
        adj_matrix = load_adjacency_matrix(matrix_path)
    except Exception as e:
        print(f"Error loading adjacency matrix: {str(e)}")
        raise

    try:
        edge_index, node_index = adjacency_to_edge_index(adj_matrix)
        print(f"Graph has {len(node_index)} nodes and {edge_index.size(1)} edges")
    except Exception as e:
        print(f"Error converting adjacency matrix to edge index: {str(e)}")
        raise

    # Train Node2Vec
    print(f"Training Node2Vec with embedding dimension {args.embedding_dim}")
    try:
        embeddings = train_node2vec(
            edge_index,
            num_nodes=len(node_index),
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            p=args.p,
            q=args.q,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_gpu=not args.cpu,
            patience=args.patience
        )
    except Exception as e:
        print(f"Error during Node2Vec training: {str(e)}")
        raise

    # Save embeddings in multiple formats
    try:
        print(f"Saving embeddings to {output_pt}")
        torch.save(embeddings, output_pt)

        # Also save as numpy array for non-PyTorch applications
        np.save(output_np, embeddings.detach().numpy())
        print(f"Saved numpy version to {output_np}")

        # Save mapping if we have named nodes
        if isinstance(adj_matrix, pd.DataFrame):
            reversed_mapping = {idx: node for node, idx in node_index.items()}
            mapping_df = pd.DataFrame(list(reversed_mapping.items()), columns=['Index', 'Node'])
            mapping_df.to_csv(mapping_output, index=False)
            print(f"Node mapping saved to {mapping_output}")

            # Save embeddings with node names as pickle
            named_embeddings = pd.DataFrame(
                embeddings.detach().numpy(),
                index=[reversed_mapping[i] for i in range(len(node_index))]
            )
            pd.to_pickle(named_embeddings, output_named)
            print(f"Named embeddings saved to {output_named}")

    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        raise

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
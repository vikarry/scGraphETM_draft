import torch
import gc
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

from sc_model import ScModel
from scDataLoader import ScDataLoader


def train_model(model: ScModel, data_loader: ScDataLoader,
                epochs: int = 20, finetune: bool = False) -> ScModel:
    """Main training function that trains the model for the specified number of epochs."""
    print(f"Starting training for {epochs} epochs...")

    train_loader = data_loader.train_loader
    test_loader = None if finetune else data_loader.test_loader

    for epoch in range(epochs):
        model.epoch = epoch + 1
        print(f"Epoch {epoch + 1}/{epochs}")

        # Calculate weights for different loss components
        recon_loss_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0.6, 8, True)
        kl_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0, 1e-2, False)
        edge_loss_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0, 10, False)

        losses = train_epoch(model, train_loader, recon_loss_weight, kl_weight, edge_loss_weight)

        # Print training information
        print(f"recon_loss_weight: {recon_loss_weight}, "
              f"kl_weight: {kl_weight}, "
              f"edge_loss_weight: {edge_loss_weight}")
        print(f"Avg Recon Loss: {losses['recon_loss']:.4f}, "
              f"Avg KL Loss: {losses['kl_loss']:.4f}, "
              f"Avg edge_recon Loss: {losses['edge_recon_loss']:.4f}, "
              f"Avg Total Loss: {losses['total_loss']:.4f}")

        if not finetune:
            model.evaluate_and_save(data_loader)

        gc.collect()

    print(f"Best Train ARI: {model.best_train_ari:.4f}, "
          f"Best Test ARI: {model.best_test_ari:.4f}")

    return model


def train_epoch(model: ScModel, train_loader,
                recon_loss_weight: float, kl_weight: float,
                edge_loss_weight: float) -> Dict[str, float]:
    """Train for a single epoch and return losses."""
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_edge_recon_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        with autocast():
            model.optimizer.zero_grad()
            recon_loss, kl_loss, edge_recon_loss, emb = model.forward(batch, train_loader)
            loss = (recon_loss_weight * recon_loss +
                    kl_loss * kl_weight +
                    edge_loss_weight * edge_recon_loss)

            model.loss_scaler.scale(loss).backward()
            model.loss_scaler.step(model.optimizer)
            model.loss_scaler.update()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_edge_recon_loss += edge_recon_loss.item()
        total_loss += loss.item()

    return {
        'recon_loss': total_recon_loss / len(train_loader),
        'kl_loss': total_kl_loss / len(train_loader),
        'edge_recon_loss': total_edge_recon_loss / len(train_loader),
        'total_loss': total_loss / len(train_loader)
    }


def train_epoch_with_scheduler(model: ScModel, data_loader: ScDataLoader,
                               epochs: int = 20, finetune: bool = False,
                               lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict:
    """
    Train with learning rate scheduler.

    Args:
        model: The ScModel instance to train
        data_loader: Data loader providing batches
        epochs: Number of epochs to train
        finetune: Whether to finetune the model (no testing)
        lr_scheduler: Optional learning rate scheduler

    Returns:
        Dictionary with training metrics
    """
    train_loader = data_loader.train_loader
    test_loader = None if finetune else data_loader.test_loader

    metrics = {
        'train_loss': [],
        'test_ari': []
    }

    for epoch in range(epochs):
        model.epoch = epoch + 1
        print(f"Epoch {epoch + 1}/{epochs}")

        # Calculate weights for different loss components
        recon_loss_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0.6, 8, True)
        kl_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0, 1e-2, False)
        edge_loss_weight = calc_weight(epoch, epochs, 0, 2 / 4, 0, 10, False)

        # Train for one epoch
        losses = train_epoch(model, train_loader, recon_loss_weight,
                             kl_weight, edge_loss_weight)

        metrics['train_loss'].append(losses['total_loss'])

        print(f"recon_loss_weight: {recon_loss_weight}, "
              f"kl_weight: {kl_weight}, "
              f"edge_loss_weight: {edge_loss_weight}")
        print(f"Avg Recon Loss: {losses['recon_loss']:.4f}, "
              f"Avg KL Loss: {losses['kl_loss']:.4f}, "
              f"Avg edge_recon Loss: {losses['edge_recon_loss']:.4f}, "
              f"Avg Total Loss: {losses['total_loss']:.4f}")

        if not finetune:
            model.evaluate_and_save(data_loader)
            metrics['test_ari'].append(model.best_test_ari)

            if lr_scheduler is not None:
                lr_scheduler.step(model.best_test_ari)
        gc.collect()

    print(f"Best Train ARI: {model.best_train_ari:.4f}, "
          f"Best Test ARI: {model.best_test_ari:.4f}")

    return metrics


def calc_weight(epoch: int, n_epochs: int, cutoff_ratio: float,
                warmup_ratio: float, min_weight: float, max_weight: float,
                reverse: bool = False) -> float:
    """Calculate weight for loss components based on training progress."""
    if epoch < n_epochs * cutoff_ratio:
        return 0.0

    fully_warmup_epoch = n_epochs * warmup_ratio
    if warmup_ratio:
        if reverse:
            if epoch < fully_warmup_epoch:
                return 1.0
            weight_progress = min(1.0, (epoch - fully_warmup_epoch) /
                                  (n_epochs - fully_warmup_epoch))
            weight = max_weight - weight_progress * (max_weight - min_weight)
        else:
            weight_progress = min(1.0, epoch / fully_warmup_epoch)
            weight = min_weight + weight_progress * (max_weight - min_weight)
        return max(min_weight, min(max_weight, weight))
    return max_weight
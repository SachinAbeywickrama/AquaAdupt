import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from models import create_models

# CONFIGURATION

DATA_PATH = "data/processed/sequences.npz"
OUTPUT_PATH = "outputs/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 48
EPOCHS = 200
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 25
WARMUP_EPOCHS = 5

PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


class CombinedLoss(nn.Module):
    def __init__(self, huber_delta=1.0, mse_weight=0.7, huber_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.huber_weight * self.huber(pred, target)


def load_data():
    print("Loading preprocessed data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.FloatTensor(data['y_single_train']),
        torch.FloatTensor(data['y_multi_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_single_val']),
        torch.FloatTensor(data['y_multi_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']),
        torch.FloatTensor(data['y_single_test']),
        torch.FloatTensor(data['y_multi_test'])
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    target_cols = list(data['target_cols']) if isinstance(data['target_cols'], np.ndarray) else data['target_cols']
    n_features = data['X_train'].shape[2]
    n_targets = len(target_cols)

    print(f"  Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    print(f"  Features: {n_features}, Targets: {n_targets} {target_cols}")
    print(f"  Sequence shape: {data['X_train'].shape}")

    return train_loader, val_loader, test_loader, n_features, n_targets, target_cols


def train_model(model, train_loader, val_loader, model_name, n_targets):
    print(f"Training {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = CombinedLoss()
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_loss = float('inf')
    best_state = None
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        lr = scheduler.step(epoch)

        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            output = model(x)

            # Weighted loss - single step more important
            loss_single = criterion(output['single_step'], y_single)
            loss_multi = criterion(output['multi_step'], y_multi)
            loss = loss_single * 1.5 + loss_multi

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr:.6f}"})

        # Validation
        model.eval()
        val_losses = []
        val_single_mse = [[] for _ in range(n_targets)]

        with torch.no_grad():
            for batch in val_loader:
                x, y_single, y_multi = [b.to(DEVICE) for b in batch]
                output = model(x)

                loss_single = criterion(output['single_step'], y_single)
                loss_multi = criterion(output['multi_step'], y_multi)
                loss = loss_single * 1.5 + loss_multi
                val_losses.append(loss.item())

                # Per-target MSE
                for t in range(n_targets):
                    mse = ((output['single_step'][:, t] - y_single[:, t]) ** 2).mean().item()
                    val_single_mse[t].append(mse)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # Per-target metrics
        target_mses = [np.mean(val_single_mse[t]) for t in range(n_targets)]
        target_str = " | ".join([f"{mse:.4f}" for mse in target_mses])

        print(f"Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Per-target MSE: [{target_str}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return best_val_loss, history


def train_ensemble(ensemble, train_loader, val_loader, n_targets):
    print("Training Meta-Ensemble")

    # Freeze base models
    for model in ensemble.base_models:
        for param in model.parameters():
            param.requires_grad = False

    meta_params = [p for n, p in ensemble.named_parameters() if 'base_models' not in n]
    print(f"  Training {sum(p.numel() for p in meta_params):,} meta-learner parameters")

    optimizer = torch.optim.Adam(meta_params, lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = CombinedLoss()
    early_stopping = EarlyStopping(patience=15)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(60):
        ensemble.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Meta Epoch {epoch + 1}/60", leave=False)
        for batch in pbar:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            output = ensemble(x)

            loss = criterion(output['single_step'], y_single) + criterion(output['multi_step'], y_multi)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        ensemble.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x, y_single, y_multi = [b.to(DEVICE) for b in batch]
                output = ensemble(x)
                loss = criterion(output['single_step'], y_single) + criterion(output['multi_step'], y_multi)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Meta Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in ensemble.state_dict().items()}

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state:
        ensemble.load_state_dict(best_state)
        ensemble.to(DEVICE)

    return best_val_loss


def evaluate(model, test_loader, target_cols):
    print("Final Evaluation")

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y_single, _ = [b.to(DEVICE) for b in batch]
            output = model(x)
            all_preds.append(output['single_step'].cpu().numpy())
            all_targets.append(y_single.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = {}

    for i, target in enumerate(target_cols):
        pred_t = preds[:, i]
        true_t = targets[:, i]

        mse = np.mean((pred_t - true_t) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_t - true_t))

        ss_res = np.sum((true_t - pred_t) ** 2)
        ss_tot = np.sum((true_t - np.mean(true_t)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        corr = np.corrcoef(pred_t, true_t)[0, 1]

        metrics[f'{target}_mse'] = float(mse)
        metrics[f'{target}_rmse'] = float(rmse)
        metrics[f'{target}_mae'] = float(mae)
        metrics[f'{target}_r2'] = float(r2)
        metrics[f'{target}_corr'] = float(corr)

        print(f"{target:12s}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}")

    # Average metrics
    avg_r2 = np.mean([metrics[f'{t}_r2'] for t in target_cols])
    avg_rmse = np.mean([metrics[f'{t}_rmse'] for t in target_cols])
    print(f"\n{'Average':12s}: R2={avg_r2:.4f}, RMSE={avg_rmse:.4f}")

    metrics['average_r2'] = float(avg_r2)
    metrics['average_rmse'] = float(avg_rmse)

    return metrics


def save_all(models, metrics, path):
    Path(path).mkdir(parents=True, exist_ok=True)

    for name in ['mrtfn', 'cnn_lstm', 'transformer']:
        torch.save(models[name].state_dict(), f"{path}/{name}.pt")
    torch.save(models['ensemble'].state_dict(), f"{path}/meta_ensemble.pt")

    with open(f"{path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels and metrics saved to {path}")


def main():
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")

    # Load data
    train_loader, val_loader, test_loader, n_features, n_targets, target_cols = load_data()

    # Create models
    models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)

    # Train each base model
    for name in ['mrtfn', 'cnn_lstm', 'transformer']:
        train_model(models[name], train_loader, val_loader, name.upper(), n_targets)

    # Train ensemble
    train_ensemble(models['ensemble'], train_loader, val_loader, n_targets)

    # Evaluate
    metrics = evaluate(models['ensemble'], test_loader, target_cols)

    # Save
    save_all(models, metrics, OUTPUT_PATH)

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()

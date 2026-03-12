import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from sklearn.model_selection import KFold

from models import create_models

# CONFIGURATION

DATA_PATH = "data/processed/sequences.npz"
NORM_PARAMS_PATH = "data/processed/norm_params.json"
OUTPUT_PATH = "outputs/overfitting"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005
PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)

    with open(NORM_PARAMS_PATH, 'r') as f:
        norm_params = json.load(f)

    target_cols = list(data['target_cols'])

    return data, target_cols, norm_params


def train_with_history(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds, train_targets = [], []

        for batch in train_loader:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output['single_step'], y_single)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(output['single_step'].detach().cpu().numpy())
            train_targets.append(y_single.cpu().numpy())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                x, y_single, y_multi = [b.to(DEVICE) for b in batch]
                output = model(x)
                loss = criterion(output['single_step'], y_single)
                val_losses.append(loss.item())
                val_preds.append(output['single_step'].cpu().numpy())
                val_targets.append(y_single.cpu().numpy())

        # Calculate R²
        train_preds_all = np.concatenate(train_preds)
        train_targets_all = np.concatenate(train_targets)
        val_preds_all = np.concatenate(val_preds)
        val_targets_all = np.concatenate(val_targets)

        train_r2 = 1 - np.sum((train_targets_all - train_preds_all) ** 2) / (
                    np.sum((train_targets_all - train_targets_all.mean()) ** 2) + 1e-8)
        val_r2 = 1 - np.sum((val_targets_all - val_preds_all) ** 2) / (
                    np.sum((val_targets_all - val_targets_all.mean()) ** 2) + 1e-8)

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        if (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch + 1}: Train Loss={np.mean(train_losses):.4f}, Val Loss={np.mean(val_losses):.4f}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")

    return history


def evaluate_on_split(model, loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]
            output = model(x)
            all_preds.append(output['single_step'].cpu().numpy())
            all_targets.append(y_single.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = np.mean((preds - targets) ** 2)
    r2 = 1 - np.sum((targets - preds) ** 2) / (np.sum((targets - targets.mean()) ** 2) + 1e-8)

    return mse, r2, preds, targets


def cross_validation(X, y_single, y_multi, n_features, n_targets, n_folds=5):
    print(f"\nPerforming {n_folds}-Fold Cross-Validation...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n  Fold {fold + 1}/{n_folds}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X[train_idx]),
            torch.FloatTensor(y_single[train_idx]),
            torch.FloatTensor(y_multi[train_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X[val_idx]),
            torch.FloatTensor(y_single[val_idx]),
            torch.FloatTensor(y_multi[val_idx])
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create fresh model
        models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)
        model = models['mrtfn']  # Use MRTFN for CV

        # Quick training (30 epochs for CV)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(30):
            model.train()
            for batch in train_loader:
                x, y_s, y_m = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out['single_step'], y_s)
                loss.backward()
                optimizer.step()

        # Evaluate
        train_mse, train_r2, _, _ = evaluate_on_split(model, train_loader)
        val_mse, val_r2, _, _ = evaluate_on_split(model, val_loader)

        fold_results.append({
            'fold': fold + 1,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'gap': train_r2 - val_r2
        })

        print(f"   Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Gap: {train_r2 - val_r2:.4f}")

    return fold_results


def plot_learning_curves(history, model_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add gap annotation
    final_train = history['train_loss'][-1]
    final_val = history['val_loss'][-1]
    gap = abs(final_val - final_train)
    ax1.annotate(f'Final Gap: {gap:.4f}', xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=11, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # R² curves
    ax2 = axes[1]
    ax2.plot(epochs, history['train_r2'], 'b-', label='Training R²', linewidth=2)
    ax2.plot(epochs, history['val_r2'], 'r-', label='Validation R²', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title(f'{model_name} - R² Score Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add convergence annotation
    final_train_r2 = history['train_r2'][-1]
    final_val_r2 = history['val_r2'][-1]
    ax2.annotate(f'Train R²: {final_train_r2:.4f}\nVal R²: {final_val_r2:.4f}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 fontsize=11, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_train_val_test_comparison(results, target_cols, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(target_cols))
    width = 0.25

    train_r2 = [results['train'][t]['r2'] for t in target_cols]
    val_r2 = [results['val'][t]['r2'] for t in target_cols]
    test_r2 = [results['test'][t]['r2'] for t in target_cols]

    bars1 = ax.bar(x - width, train_r2, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, val_r2, width, label='Validation', color='orange', alpha=0.8)
    bars3 = ax.bar(x + width, test_r2, width, label='Test', color='green', alpha=0.8)

    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Train vs Validation vs Test Performance\n(Similar performance = No Overfitting)', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in target_cols], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_path}/train_val_test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_results(cv_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    folds = [r['fold'] for r in cv_results]
    train_r2 = [r['train_r2'] for r in cv_results]
    val_r2 = [r['val_r2'] for r in cv_results]
    gaps = [r['gap'] for r in cv_results]

    # R² per fold
    ax1 = axes[0]
    x = np.arange(len(folds))
    width = 0.35
    ax1.bar(x - width / 2, train_r2, width, label='Train R²', color='steelblue', alpha=0.8)
    ax1.bar(x + width / 2, val_r2, width, label='Val R²', color='orange', alpha=0.8)
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add mean annotation
    mean_train = np.mean(train_r2)
    mean_val = np.mean(val_r2)
    ax1.axhline(mean_train, color='steelblue', linestyle='--', alpha=0.5)
    ax1.axhline(mean_val, color='orange', linestyle='--', alpha=0.5)
    ax1.annotate(f'Mean Train: {mean_train:.4f}\nMean Val: {mean_val:.4f}',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Generalization gap
    ax2 = axes[1]
    colors = ['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gaps]
    ax2.bar(folds, gaps, color=colors, alpha=0.8)
    ax2.axhline(0.05, color='green', linestyle='--', label='Good (<0.05)', alpha=0.7)
    ax2.axhline(0.1, color='red', linestyle='--', label='Warning (>0.1)', alpha=0.7)
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Generalization Gap (Train R² - Val R²)', fontsize=12)
    ax2.set_title('Generalization Gap per Fold\n(Lower = Less Overfitting)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    mean_gap = np.mean(gaps)
    ax2.annotate(f'Mean Gap: {mean_gap:.4f}',
                 xy=(0.98, 0.98), xycoords='axes fraction',
                 fontsize=11, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen' if mean_gap < 0.05 else 'wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_path}/cross_validation_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_summary(results, cv_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'OVERFITTING ANALYSIS SUMMARY', fontsize=18, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Key metrics
    train_r2 = np.mean([results['train'][t]['r2'] for t in results['train']])
    val_r2 = np.mean([results['val'][t]['r2'] for t in results['val']])
    test_r2 = np.mean([results['test'][t]['r2'] for t in results['test']])

    cv_mean_gap = np.mean([r['gap'] for r in cv_results])
    cv_std_gap = np.std([r['gap'] for r in cv_results])

    summary_text = f"""
  
    
    1. TRAIN/VAL/TEST PERFORMANCE 

       Training R²:     {train_r2:.4f}
       Validation R²:   {val_r2:.4f}
       Test R²:         {test_r2:.4f}
       
       Train-Test Gap:  {abs(train_r2 - test_r2):.4f}  {'GOOD (<0.05)' if abs(train_r2 - test_r2) < 0.05 else ' WARNING'}
    

    
    2. CROSS-VALIDATION RESULTS (5-Fold)

       Mean Generalization Gap: {cv_mean_gap:.4f} ± {cv_std_gap:.4f}
       
       Interpretation: {' NO OVERFITTING' if cv_mean_gap < 0.05 else 'MILD OVERFITTING' if cv_mean_gap < 0.1 else 'OVERFITTING'}


    """

    ax.text(0.5, 0.45, summary_text, fontsize=11, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.savefig(f"{output_path}/overfitting_summary.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    print("AQUAADAPT - OVERFITTING ANALYSIS")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data, target_cols, norm_params = load_data()

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_single_train = data['y_single_train']
    y_single_val = data['y_single_val']
    y_single_test = data['y_single_test']
    y_multi_train = data['y_multi_train']
    y_multi_val = data['y_multi_val']
    y_multi_test = data['y_multi_test']

    n_features = X_train.shape[2]
    n_targets = len(target_cols)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_single_train),
        torch.FloatTensor(y_multi_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_single_val),
        torch.FloatTensor(y_multi_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_single_test),
        torch.FloatTensor(y_multi_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. LEARNING CURVES

    print("1. GENERATING LEARNING CURVES")

    models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)

    print("\nTraining MRTFN with learning curve tracking...")
    history = train_with_history(models['mrtfn'], train_loader, val_loader, epochs=EPOCHS)
    plot_learning_curves(history, 'MRTFN', OUTPUT_PATH)

    # 2. TRAIN/VAL/TEST COMPARISON

    print("2. TRAIN/VAL/TEST PERFORMANCE COMPARISON")

    results = {'train': {}, 'val': {}, 'test': {}}

    # Evaluate on all splits
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        mse, r2, preds, targets = evaluate_on_split(models['mrtfn'], loader)

        for i, target in enumerate(target_cols):
            t_preds = preds[:, i]
            t_targets = targets[:, i]
            t_r2 = 1 - np.sum((t_targets - t_preds) ** 2) / (np.sum((t_targets - t_targets.mean()) ** 2) + 1e-8)
            results[split_name][target] = {'r2': t_r2}

        print(f"\n{split_name.upper()} R²: {r2:.4f}")
        for target in target_cols:
            print(f"  {target}: {results[split_name][target]['r2']:.4f}")

    plot_train_val_test_comparison(results, target_cols, OUTPUT_PATH)

    # 3. CROSS-VALIDATION

    print("3. CROSS-VALIDATION ANALYSIS")

    # Combine train + val for CV
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_single_combined = np.concatenate([y_single_train, y_single_val], axis=0)
    y_multi_combined = np.concatenate([y_multi_train, y_multi_val], axis=0)

    cv_results = cross_validation(X_combined, y_single_combined, y_multi_combined,
                                  n_features, n_targets, n_folds=5)

    plot_cv_results(cv_results, OUTPUT_PATH)

    # 4. SUMMARY

    print("4. GENERATING SUMMARY")

    plot_overfitting_summary(results, cv_results, OUTPUT_PATH)

    # Save results
    analysis_results = {
        'learning_curves': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_r2': history['train_r2'][-1],
            'final_val_r2': history['val_r2'][-1],
            'loss_gap': abs(history['val_loss'][-1] - history['train_loss'][-1])
        },
        'split_comparison': {
            'train_avg_r2': np.mean([results['train'][t]['r2'] for t in target_cols]),
            'val_avg_r2': np.mean([results['val'][t]['r2'] for t in target_cols]),
            'test_avg_r2': np.mean([results['test'][t]['r2'] for t in target_cols])
        },
        'cross_validation': {
            'mean_train_r2': np.mean([r['train_r2'] for r in cv_results]),
            'mean_val_r2': np.mean([r['val_r2'] for r in cv_results]),
            'mean_gap': np.mean([r['gap'] for r in cv_results]),
            'std_gap': np.std([r['gap'] for r in cv_results])
        }
    }

    with open(f"{OUTPUT_PATH}/overfitting_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)

    # Print summary

    print("OVERFITTING ANALYSIS COMPLETE")

    print("\nKEY FINDINGS:")
    print(f"  Train R²: {analysis_results['split_comparison']['train_avg_r2']:.4f}")
    print(f"  Val R²:   {analysis_results['split_comparison']['val_avg_r2']:.4f}")
    print(f"  Test R²:  {analysis_results['split_comparison']['test_avg_r2']:.4f}")
    print(
        f"  Train-Test Gap: {abs(analysis_results['split_comparison']['train_avg_r2'] - analysis_results['split_comparison']['test_avg_r2']):.4f}")
    print(
        f"\n  CV Mean Gap: {analysis_results['cross_validation']['mean_gap']:.4f} ± {analysis_results['cross_validation']['std_gap']:.4f}")

    gap = abs(
        analysis_results['split_comparison']['train_avg_r2'] - analysis_results['split_comparison']['test_avg_r2'])
    if gap < 0.05:
        print("\n   CONCLUSION: Model is NOT overfitting")
    elif gap < 0.1:
        print("\n   CONCLUSION: Mild overfitting, but acceptable")
    else:
        print("\n   CONCLUSION: Model may be overfitting")


if __name__ == "__main__":
    main()

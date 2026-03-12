import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from models import create_models

# CONFIGURATION

DATA_PATH = "data/processed/sequences.npz"
NORM_PARAMS_PATH = "data/processed/norm_params.json"
MODELS_PATH = "outputs/"
OUTPUT_PATH = "outputs/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]
HORIZON_LABELS = ['1h', '2h', '6h', '12h', '24h', '48h']


def load_data_and_models():
    print("Loading data and models...")

    data = np.load(DATA_PATH, allow_pickle=True)

    # Load normalization params for denormalization
    with open(NORM_PARAMS_PATH, 'r') as f:
        norm_params = json.load(f)

    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']),
        torch.FloatTensor(data['y_single_test']),
        torch.FloatTensor(data['y_multi_test'])
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    target_cols = list(data['target_cols']) if isinstance(data['target_cols'], np.ndarray) else list(
        data['target_cols'])
    n_features = data['X_test'].shape[2]
    n_targets = len(target_cols)

    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Targets: {target_cols}")

    # Load models
    models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)

    for name in ['mrtfn', 'cnn_lstm', 'transformer']:
        path = f"{MODELS_PATH}/{name}.pt"
        if Path(path).exists():
            models[name].load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"  Loaded {name}")

    ensemble_path = f"{MODELS_PATH}/meta_ensemble.pt"
    if Path(ensemble_path).exists():
        models['ensemble'].load_state_dict(torch.load(ensemble_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded ensemble")

    return test_loader, models, target_cols, norm_params, data['y_single_test'], data['y_multi_test']


def compute_metrics(preds, targets, target_cols, norm_params):
    metrics = {}
    y_std = np.array(norm_params['y_single_std_raw'])
    y_mean = np.array(norm_params['y_single_mean_raw'])

    for i, target in enumerate(target_cols):
        pred_t = preds[:, i]
        true_t = targets[:, i]

        # Basic metrics
        mse = np.mean((pred_t - true_t) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_t - true_t))

        # R² score
        ss_res = np.sum((true_t - pred_t) ** 2)
        ss_tot = np.sum((true_t - np.mean(true_t)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Correlation
        corr = np.corrcoef(pred_t, true_t)[0, 1]

        # Denormalize for interpretable metrics
        pred_denorm = pred_t * y_std[i] + y_mean[i]
        true_denorm = true_t * y_std[i] + y_mean[i]

        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        mask = np.abs(true_denorm) > 0.01
        if mask.sum() > 0:
            mape = np.mean(np.abs((true_denorm[mask] - pred_denorm[mask]) / true_denorm[mask])) * 100
        else:
            mape = 0.0

        # Normalized RMSE (as percentage of mean)
        nrmse = (rmse * y_std[i]) / (y_mean[i] + 1e-8) * 100

        # Normalized MAE (as percentage of mean)
        nmae = (mae * y_std[i]) / (y_mean[i] + 1e-8) * 100

        # Accuracy
        accuracy = max(0, 100 - nrmse)

        metrics[target] = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'Correlation': float(corr),
            'MAPE': float(mape),
            'NRMSE_percent': float(nrmse),
            'NMAE_percent': float(nmae),
            'Accuracy_percent': float(accuracy),
            'Mean_actual': float(y_mean[i]),
            'Std_actual': float(y_std[i])
        }

    return metrics


def evaluate_model(model, test_loader, target_cols, norm_params, model_name):
    model.eval()
    all_preds_single, all_targets_single = [], []
    all_preds_multi, all_targets_multi = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]
            output = model(x)

            all_preds_single.append(output['single_step'].cpu().numpy())
            all_targets_single.append(y_single.cpu().numpy())
            all_preds_multi.append(output['multi_step'].cpu().numpy())
            all_targets_multi.append(y_multi.cpu().numpy())

    preds_single = np.concatenate(all_preds_single)
    targets_single = np.concatenate(all_targets_single)
    preds_multi = np.concatenate(all_preds_multi)
    targets_multi = np.concatenate(all_targets_multi)

    # Single-step metrics
    single_metrics = compute_metrics(preds_single, targets_single, target_cols, norm_params)

    # Multi-step metrics by horizon
    horizon_metrics = {}
    for h_idx, (horizon, label) in enumerate(zip(PREDICTION_HORIZONS, HORIZON_LABELS)):
        h_preds = preds_multi[:, h_idx, :]
        h_targets = targets_multi[:, h_idx, :]
        h_mse = np.mean((h_preds - h_targets) ** 2)

        # Per-target R² at this horizon
        r2_per_target = {}
        for t_idx, target in enumerate(target_cols):
            ss_res = np.sum((h_targets[:, t_idx] - h_preds[:, t_idx]) ** 2)
            ss_tot = np.sum((h_targets[:, t_idx] - np.mean(h_targets[:, t_idx])) ** 2)
            r2_per_target[target] = float(1 - ss_res / (ss_tot + 1e-8))

        horizon_metrics[label] = {
            'horizon_steps': horizon,
            'horizon_minutes': horizon * 20,
            'overall_mse': float(h_mse),
            'r2_per_target': r2_per_target
        }

    return {
        'single_step_metrics': single_metrics,
        'horizon_metrics': horizon_metrics,
        'predictions': preds_single,
        'targets': targets_single
    }


def plot_results(results, target_cols, model_name, output_path, norm_params):
    preds = results['predictions']
    targets = results['targets']
    metrics = results['single_step_metrics']

    n_targets = len(target_cols)
    fig, axes = plt.subplots(3, n_targets, figsize=(5 * n_targets, 12))

    y_std = np.array(norm_params['y_single_std_raw'])
    y_mean = np.array(norm_params['y_single_mean_raw'])

    for i, target in enumerate(target_cols):
        r2 = metrics[target]['R2']
        acc = metrics[target]['Accuracy_percent']

        # Denormalize for plotting
        pred_denorm = preds[:, i] * y_std[i] + y_mean[i]
        true_denorm = targets[:, i] * y_std[i] + y_mean[i]

        # Row 1: Scatter plot
        ax1 = axes[0, i]
        ax1.scatter(true_denorm[:500], pred_denorm[:500], alpha=0.5, s=15, c='steelblue')
        min_val = min(true_denorm.min(), pred_denorm.min())
        max_val = max(true_denorm.max(), pred_denorm.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Actual', fontsize=10)
        ax1.set_ylabel('Predicted', fontsize=10)
        ax1.set_title(f'{target.upper()}\nR²={r2:.3f}, Acc={acc:.1f}%', fontsize=11, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Row 2: Time series (first 300 points)
        ax2 = axes[1, i]
        t = np.arange(min(300, len(true_denorm)))
        ax2.plot(t, true_denorm[:300], label='Actual', alpha=0.8, linewidth=1.5)
        ax2.plot(t, pred_denorm[:300], label='Predicted', alpha=0.8, linewidth=1.5)
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel(f'{target.upper()} Value', fontsize=10)
        ax2.set_title('Time Series Comparison', fontsize=10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Row 3: Error distribution
        ax3 = axes[2, i]
        errors = pred_denorm - true_denorm
        ax3.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean={errors.mean():.3f}')
        ax3.set_xlabel('Prediction Error', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title(f'Error Distribution (Std={errors.std():.3f})', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_comparison(all_results, target_cols, output_path):
    fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]

    colors = {'mrtfn': 'tab:blue', 'cnn_lstm': 'tab:orange', 'transformer': 'tab:green', 'ensemble': 'tab:red'}

    for t_idx, target in enumerate(target_cols):
        ax = axes[t_idx]

        for model_name in ['mrtfn', 'cnn_lstm', 'transformer', 'ensemble']:
            if model_name in all_results:
                r2_values = []
                for label in HORIZON_LABELS:
                    r2 = all_results[model_name]['horizon_metrics'][label]['r2_per_target'][target]
                    r2_values.append(r2)

                ax.plot(HORIZON_LABELS, r2_values, 'o-', label=model_name.upper(),
                        color=colors[model_name], linewidth=2, markersize=6)

        ax.set_xlabel('Prediction Horizon', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title(f'{target.upper()} - Multi-Horizon Performance', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{output_path}/horizon_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(all_results, target_cols):
    print("PERFORMANCE SUMMARY")

    # Table header
    header = f"{'Model':<15}" + "".join([f"{t.upper():>15}" for t in target_cols]) + f"{'AVERAGE':>15}"
    print("\n" + header)
    print("-" * len(header))

    # R² scores
    print("\nR² Score:")
    for model_name in ['mrtfn', 'cnn_lstm', 'transformer', 'ensemble']:
        if model_name in all_results:
            row = f"{model_name.upper():<15}"
            r2_values = []
            for target in target_cols:
                r2 = all_results[model_name]['single_step_metrics'][target]['R2']
                r2_values.append(r2)
                row += f"{r2:>15.4f}"
            row += f"{np.mean(r2_values):>15.4f}"
            print(row)

    # Accuracy
    print("\nAccuracy % :")
    for model_name in ['mrtfn', 'cnn_lstm', 'transformer', 'ensemble']:
        if model_name in all_results:
            row = f"{model_name.upper():<15}"
            acc_values = []
            for target in target_cols:
                acc = all_results[model_name]['single_step_metrics'][target]['Accuracy_percent']
                acc_values.append(acc)
                row += f"{acc:>14.1f}%"
            row += f"{np.mean(acc_values):>14.1f}%"
            print(row)

    # MAPE
    print("\nMAPE % :")
    for model_name in ['mrtfn', 'cnn_lstm', 'transformer', 'ensemble']:
        if model_name in all_results:
            row = f"{model_name.upper():<15}"
            mape_values = []
            for target in target_cols:
                mape = all_results[model_name]['single_step_metrics'][target]['MAPE']
                mape_values.append(mape)
                row += f"{mape:>14.1f}%"
            row += f"{np.mean(mape_values):>14.1f}%"
            print(row)


def main():
    print("AQUAADAPT - COMPREHENSIVE EVALUATION")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    test_loader, models, target_cols, norm_params, y_test_single, y_test_multi = load_data_and_models()

    all_results = {}

    for model_name in ['mrtfn', 'cnn_lstm', 'transformer', 'ensemble']:
        print(f"\nEvaluating {model_name.upper()}...")

        results = evaluate_model(models[model_name], test_loader, target_cols, norm_params, model_name)
        all_results[model_name] = {
            'single_step_metrics': results['single_step_metrics'],
            'horizon_metrics': results['horizon_metrics']
        }

        # Generate plots
        plot_results(results, target_cols, model_name, OUTPUT_PATH, norm_params)

    # Horizon comparison plot
    plot_horizon_comparison(all_results, target_cols, OUTPUT_PATH)

    # Save all results
    with open(f"{OUTPUT_PATH}/evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary_table(all_results, target_cols)

    # Best model summary
    print("\nEnsemble Final Metrics:")
    for target in target_cols:
        m = all_results['ensemble']['single_step_metrics'][target]
        print(f"{target.upper():12s}: R²={m['R2']:.4f}, Accuracy={m['Accuracy_percent']:.1f}%, MAPE={m['MAPE']:.1f}%")

    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()

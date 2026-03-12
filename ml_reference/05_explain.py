
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_models


# CONFIGURATION

DATA_PATH = "data/processed/sequences.npz"
MODELS_PATH = "outputs/"
OUTPUT_PATH = "outputs/explanations/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]
N_SAMPLES = 100


def load_data_and_model():
    
    print("Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    
    X_test = data['X_test'][:N_SAMPLES]
    
    feature_cols = data['feature_cols']
    target_cols = data['target_cols']
    
    if isinstance(feature_cols, np.ndarray):
        feature_cols = feature_cols.tolist()
    if isinstance(target_cols, np.ndarray):
        target_cols = target_cols.tolist()
    
    n_features = data['X_test'].shape[2]
    n_targets = len(target_cols)
    
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    print(f"  Targets: {target_cols}")
    
    # Load ensemble model
    print("\nLoading ensemble model...")
    models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)
    
    ensemble_path = f"{MODELS_PATH}/meta_ensemble.pt"
    if Path(ensemble_path).exists():
        models['ensemble'].load_state_dict(torch.load(ensemble_path, map_location=DEVICE))
        print("  Loaded ensemble")
    
    return X_test, models['ensemble'], feature_cols, target_cols


def compute_permutation_importance(model, X, feature_cols, target_cols):
    
    print("\nComputing feature importance (permutation method)...")
    
    model.eval()
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Baseline prediction
    with torch.no_grad():
        baseline_output = model(X_tensor)
        baseline_pred = baseline_output['single_step'].cpu().numpy()
    
    baseline_mse = np.mean(baseline_pred ** 2, axis=0)
    
    # Compute importance for each feature
    importance = {target: [] for target in target_cols}
    feature_names_used = []
    
    for f_idx in range(X.shape[2]):
        if f_idx >= len(feature_cols):
            fname = f'feature_{f_idx}'
        else:
            fname = feature_cols[f_idx]
        
        feature_names_used.append(fname)
        
        # Permute feature across all time steps
        X_permuted = X.copy()
        for t in range(X.shape[1]):
            np.random.shuffle(X_permuted[:, t, f_idx])
        
        X_perm_tensor = torch.FloatTensor(X_permuted).to(DEVICE)
        
        with torch.no_grad():
            perm_output = model(X_perm_tensor)
            perm_pred = perm_output['single_step'].cpu().numpy()
        
        perm_mse = np.mean(perm_pred ** 2, axis=0)
        
        # Importance = increase in MSE when feature is permuted
        imp = perm_mse - baseline_mse
        
        for i, target in enumerate(target_cols):
            importance[target].append(float(imp[i]) if imp.ndim > 0 else float(imp))
        
        if (f_idx + 1) % 20 == 0:
            print(f"  Processed {f_idx + 1}/{X.shape[2]} features...")
    
    return importance, feature_names_used


def compute_temporal_importance(model, X, target_cols):
    
    print("\nComputing temporal importance...")
    
    model.eval()
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Baseline
    with torch.no_grad():
        baseline_pred = model(X_tensor)['single_step'].cpu().numpy()
    baseline_mse = np.mean(baseline_pred ** 2, axis=0)
    
    seq_len = X.shape[1]
    temporal_importance = np.zeros((seq_len, len(target_cols)))
    
    for t in range(seq_len):
        X_permuted = X.copy()
        perm = np.random.permutation(len(X))
        X_permuted[:, t, :] = X_permuted[perm, t, :]
        
        X_perm_tensor = torch.FloatTensor(X_permuted).to(DEVICE)
        
        with torch.no_grad():
            perm_pred = model(X_perm_tensor)['single_step'].cpu().numpy()
        
        perm_mse = np.mean(perm_pred ** 2, axis=0)
        temporal_importance[t] = perm_mse - baseline_mse
        
        if (t + 1) % 20 == 0:
            print(f"  Processed {t + 1}/{seq_len} time steps...")
    
    return temporal_importance


def get_attention_weights(model, X):
    
    print("\nExtracting attention weights...")
    
    model.eval()
    X_tensor = torch.FloatTensor(X[:10]).to(DEVICE)
    
    try:
        mrtfn = model.base_models[0]
        with torch.no_grad():
            output = mrtfn(X_tensor, return_attention=True)
        
        if 'attention_weights' in output and output['attention_weights'] is not None:
            attn = output['attention_weights'].cpu().numpy()
            print(f"  Attention shape: {attn.shape}")
            return attn
    except Exception as e:
        print(f"  Could not extract attention: {e}")
    
    return None


def plot_feature_importance(importance, feature_names, target, top_k=20, output_path=None):
    
    values = importance[target]
    sorted_idx = np.argsort(values)[-top_k:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] if i < len(feature_names) else f'f_{i}' for i in sorted_idx],
        [values[i] for i in sorted_idx],
        color='steelblue'
    )
    plt.xlabel('Importance Score (MSE increase)')
    plt.title(f'Top {top_k} Features for {target} Prediction')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path}/feature_importance_{target}.png", dpi=150)
        print(f"  Saved: feature_importance_{target}.png")
    
    plt.close()


def plot_temporal_importance(temporal_importance, target_cols, output_path=None):

    seq_len = temporal_importance.shape[0]
    
    plt.figure(figsize=(14, 6))
    
    for i, target in enumerate(target_cols):
        plt.plot(range(seq_len), temporal_importance[:, i], label=target, linewidth=2)
    
    plt.xlabel('Time Step (older -> newer)')
    plt.ylabel('Importance Score')
    plt.title('Temporal Importance: Which Time Steps Matter Most')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    n_labels = 8
    step = seq_len // n_labels
    time_labels = [f't-{(seq_len-i-1)*20}min' for i in range(0, seq_len, step)]
    plt.xticks(range(0, seq_len, step), time_labels, rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path}/temporal_importance.png", dpi=150)
    
    plt.close()


def plot_attention_heatmap(attention, output_path=None):

    if attention is None:
        return
    
    attn = attention[0]
    if attn.ndim == 3:
        attn = attn.mean(axis=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn, cmap='Blues', xticklabels=10, yticklabels=10)
    plt.xlabel('Key Position (Time Steps)')
    plt.ylabel('Query Position (Time Steps)')
    plt.title('Temporal Self-Attention Weights')
    
    if output_path:
        plt.savefig(f"{output_path}/attention_heatmap.png", dpi=150)
    
    plt.close()


def explain_single_prediction(model, x, feature_cols, target_cols, top_k=10):
    
    model.eval()
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
    x_tensor.requires_grad = True
    
    output = model(x_tensor)
    pred = output['single_step'][0]
    
    explanation = {
        'prediction': {target: float(pred[i].item()) for i, target in enumerate(target_cols)},
        'top_features': {}
    }
    
    for i, target in enumerate(target_cols):
        model.zero_grad()
        if x_tensor.grad is not None:
            x_tensor.grad.zero_()
        
        pred[i].backward(retain_graph=True)
        
        if x_tensor.grad is not None:
            gradients = x_tensor.grad.abs().cpu().numpy()[0]
            feature_importance = gradients.mean(axis=0)
            
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            
            explanation['top_features'][target] = [
                {
                    'feature': feature_cols[idx] if idx < len(feature_cols) else f'feature_{idx}',
                    'importance': float(feature_importance[idx])
                }
                for idx in top_indices
            ]
    
    return explanation


def main():
    print("SHAP EXPLAINABILITY ANALYSIS")
    print(f"Device: {DEVICE}")
    
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    X_test, ensemble, feature_cols, target_cols = load_data_and_model()
    
    # 1. Feature importance
    importance, feature_names = compute_permutation_importance(
        ensemble, X_test, feature_cols, target_cols
    )
    
    print("\nGenerating feature importance plots...")
    for target in target_cols:
        plot_feature_importance(importance, feature_names, target, top_k=20, output_path=OUTPUT_PATH)
    
    # 2. Temporal importance
    temporal_imp = compute_temporal_importance(ensemble, X_test, target_cols)
    plot_temporal_importance(temporal_imp, target_cols, output_path=OUTPUT_PATH)
    
    # 3. Attention weights
    attention = get_attention_weights(ensemble, X_test)
    if attention is not None:
        plot_attention_heatmap(attention, output_path=OUTPUT_PATH)
    
    # 4. Sample explanations
    print("\nGenerating sample explanations...")
    sample_explanations = []
    for i in range(min(5, len(X_test))):
        exp = explain_single_prediction(ensemble, X_test[i], feature_cols, target_cols)
        sample_explanations.append(exp)
    
    # Save results
    results = {
        'feature_importance': importance,
        'temporal_importance': temporal_imp.tolist(),
        'sample_explanations': sample_explanations
    }
    
    with open(f"{OUTPUT_PATH}/explanations.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("TOP 10 FEATURES BY TARGET")
    
    for target in target_cols:
        print(f"\n{target}:")
        values = importance[target]
        sorted_idx = np.argsort(values)[-10:][::-1]
        for rank, idx in enumerate(sorted_idx, 1):
            fname = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
            print(f"  {rank:2d}. {fname:30s}: {values[idx]:.4f}")

    print("EXPLAINABILITY ANALYSIS COMPLETE")



if __name__ == "__main__":
    main()

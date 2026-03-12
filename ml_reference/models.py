
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    # Residual block for better gradient flow
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))


class TemporalAttention(nn.Module):
    # Multi-head temporal attention
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(x), attn


class MRTFN(nn.Module):
    # Multi-Resolution Temporal Fusion Network - Final Version
    
    def __init__(self, n_features, n_targets, hidden_dim=256, num_layers=3, dropout=0.15,
                 prediction_horizons=[3, 6, 18, 36, 72, 144]):
        super().__init__()
        
        self.n_targets = n_targets
        self.prediction_horizons = prediction_horizons
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal attention
        self.attention = TemporalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Residual refinement
        self.refine = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        
        # Output heads with skip connection
        self.single_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concat of attention + last hidden
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_targets)
        )
        
        self.multi_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_targets)
            ) for _ in prediction_horizons
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, return_attention=False):
        # Input projection
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Attention over LSTM outputs
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Refine with residual blocks
        refined = self.refine(attn_out)
        
        # Global representation: attention pooling + last hidden state
        attn_pooled = refined.mean(dim=1)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Bidirectional
        
        global_repr = torch.cat([attn_pooled, last_hidden], dim=1)
        
        # Predictions
        single_pred = self.single_head(global_repr)
        multi_preds = torch.stack([head(global_repr) for head in self.multi_heads], dim=1)
        
        output = {'single_step': single_pred, 'multi_step': multi_preds}
        if return_attention:
            output['attention_weights'] = attn_weights
        return output


class CNNLSTMModel(nn.Module):
    # CNN-LSTM with dilated convolutions for multi-scale patterns
    
    def __init__(self, n_features, n_targets, hidden_dim=256, dropout=0.15,
                 prediction_horizons=[3, 6, 18, 36, 72, 144]):
        super().__init__()
        
        self.prediction_horizons = prediction_horizons
        
        # Multi-scale CNN with dilated convolutions
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm1d(64),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(128),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(128),
                nn.GELU()
            )
        ])
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # LSTM
        self.lstm = nn.LSTM(128, hidden_dim // 2, num_layers=2, dropout=dropout,
                           batch_first=True, bidirectional=True)
        
        # Output heads
        self.single_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_targets)
        )
        
        self.multi_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_targets)
            ) for _ in prediction_horizons
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # CNN: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        
        for conv in self.conv_blocks:
            x = conv(x)
        
        # Channel attention
        attn = self.channel_attn(x).unsqueeze(-1)
        x = x * attn
        
        # Back to (batch, seq, features)
        x = x.transpose(1, 2)
        
        # LSTM
        _, (h_n, _) = self.lstm(x)
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        single_pred = self.single_head(final)
        multi_preds = torch.stack([head(final) for head in self.multi_heads], dim=1)
        
        return {'single_step': single_pred, 'multi_step': multi_preds}


class TransformerModel(nn.Module):
    # Transformer with relative positional encoding
    
    def __init__(self, n_features, n_targets, d_model=256, nhead=8, num_layers=4, dropout=0.15,
                 prediction_horizons=[3, 6, 18, 36, 72, 144]):
        super().__init__()
        
        self.prediction_horizons = prediction_horizons
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection with residual refinement
        self.output_refine = nn.Sequential(
            ResidualBlock(d_model, dropout),
            ResidualBlock(d_model, dropout)
        )
        
        self.single_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, n_targets)
        )
        
        self.multi_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_targets)
            ) for _ in prediction_horizons
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer
        encoded = self.transformer(x)
        
        # Refine
        refined = self.output_refine(encoded)
        
        # Global representation (mean pooling + last token)
        global_repr = refined.mean(dim=1) + refined[:, -1, :]
        
        single_pred = self.single_head(global_repr)
        multi_preds = torch.stack([head(global_repr) for head in self.multi_heads], dim=1)
        
        return {'single_step': single_pred, 'multi_step': multi_preds}


class MetaEnsemble(nn.Module):
    # self-optimizing ensemble with learned model weights
    
    def __init__(self, base_models, n_features, n_targets, n_horizons):
        super().__init__()
        
        self.base_models = base_models
        self.n_models = len(base_models)
        self.n_targets = n_targets
        self.n_horizons = n_horizons
        
        # Per-target meta-learner (different weights for different targets)
        self.target_meta = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features * 2, 64),  # mean + std of input
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, self.n_models),
                nn.Softmax(dim=-1)
            ) for _ in range(n_targets)
        ])
        
        # Global meta-learner for multi-step
        self.global_meta = nn.Sequential(
            nn.Linear(n_features * 2, 64),
            nn.GELU(),
            nn.Linear(64, self.n_models),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_weights=False):
        # Get predictions from all base models
        preds_single = []
        preds_multi = []
        
        for model in self.base_models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                out = model(x)
                preds_single.append(out['single_step'])
                preds_multi.append(out['multi_step'])
        
        # Stack: (batch, n_models, n_targets)
        single_stack = torch.stack(preds_single, dim=1)
        multi_stack = torch.stack(preds_multi, dim=1)
        
        # Input statistics for meta-learner
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1)
        stats = torch.cat([x_mean, x_std], dim=-1)
        
        # Per-target weighted ensemble
        ensemble_single = []
        all_weights = []
        
        for t in range(self.n_targets):
            weights = self.target_meta[t](stats)  # (batch, n_models)
            all_weights.append(weights)
            target_preds = single_stack[:, :, t]  # (batch, n_models)
            weighted = (target_preds * weights).sum(dim=1)  # (batch,)
            ensemble_single.append(weighted)
        
        ensemble_single = torch.stack(ensemble_single, dim=1)  # (batch, n_targets)
        
        # Global weights for multi-step
        global_weights = self.global_meta(stats)
        global_weights = global_weights.view(-1, self.n_models, 1, 1)
        ensemble_multi = (multi_stack * global_weights).sum(dim=1)
        
        output = {'single_step': ensemble_single, 'multi_step': ensemble_multi}
        if return_weights:
            output['weights'] = torch.stack(all_weights, dim=-1)  # (batch, n_models, n_targets)
        return output


def create_models(n_features, n_targets, prediction_horizons, device='cpu'):

    print(f"Creating optimized models: {n_features} features, {n_targets} targets")
    
    mrtfn = MRTFN(n_features, n_targets, prediction_horizons=prediction_horizons).to(device)
    cnn_lstm = CNNLSTMModel(n_features, n_targets, prediction_horizons=prediction_horizons).to(device)
    transformer = TransformerModel(n_features, n_targets, prediction_horizons=prediction_horizons).to(device)
    
    base_models = nn.ModuleList([mrtfn, cnn_lstm, transformer])
    
    ensemble = MetaEnsemble(
        base_models=base_models,
        n_features=n_features,
        n_targets=n_targets,
        n_horizons=len(prediction_horizons)
    ).to(device)
    
    # Print model sizes
    for name, model in [('MRTFN', mrtfn), ('CNN-LSTM', cnn_lstm), ('Transformer', transformer)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")
    
    return {
        'mrtfn': mrtfn,
        'cnn_lstm': cnn_lstm,
        'transformer': transformer,
        'ensemble': ensemble
    }

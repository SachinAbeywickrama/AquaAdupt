import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from collections import deque
import random
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from models import create_models

# CONFIGURATION

DATA_PATH = "data/processed/sequences.npz"
NORM_PARAMS_PATH = "data/processed/norm_params.json"
MODELS_PATH = "outputs/"
OUTPUT_PATH = "outputs/rl/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# RL parameters
N_EPISODES = 100
BATCH_SIZE = 32
RL_LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]
TARGET_COLS = ['ph', 'water_temp', 'turbidity']


# RL COMPONENTS


class RetrainingState:

    def __init__(self, state_dim: int = 20):
        self.state_dim = state_dim
        self.prediction_errors = deque(maxlen=100)
        self.data_drift_scores = deque(maxlen=50)
        self.model_confidences = {'mrtfn': 1.0, 'cnn_lstm': 1.0, 'transformer': 1.0}
        self.time_since_retrain = {'mrtfn': 0, 'cnn_lstm': 0, 'transformer': 0}
        self.recent_accuracy = deque(maxlen=20)
        self.per_target_errors = {t: deque(maxlen=20) for t in TARGET_COLS}

    def update(self, error, drift, confidences, accuracy, target_errors=None):
        self.prediction_errors.append(error)
        self.data_drift_scores.append(drift)
        self.recent_accuracy.append(accuracy)

        for name, conf in confidences.items():
            self.model_confidences[name] = conf

        for name in self.time_since_retrain:
            self.time_since_retrain[name] += 1

        if target_errors:
            for t, err in target_errors.items():
                if t in self.per_target_errors:
                    self.per_target_errors[t].append(err)

    def reset_retrain_time(self, model_name):
        if model_name == 'all':
            for name in self.time_since_retrain:
                self.time_since_retrain[name] = 0
        elif model_name in self.time_since_retrain:
            self.time_since_retrain[model_name] = 0

    def to_tensor(self):
        state = []

        # Error statistics
        errors = list(self.prediction_errors) if self.prediction_errors else [0.5]
        state.extend([
            np.mean(errors),
            np.std(errors) if len(errors) > 1 else 0,
            np.max(errors),
            errors[-1]
        ])

        # Drift statistics
        drifts = list(self.data_drift_scores) if self.data_drift_scores else [0]
        state.extend([np.mean(drifts), np.max(drifts)])

        # Model confidences and time since retrain
        for name in ['mrtfn', 'cnn_lstm', 'transformer']:
            state.append(self.model_confidences.get(name, 1.0))
            state.append(min(self.time_since_retrain.get(name, 0) / 100, 1.0))

        # Accuracy statistics
        accuracies = list(self.recent_accuracy) if self.recent_accuracy else [0.5]
        state.extend([np.mean(accuracies), accuracies[-1]])

        # Per-target error trends
        for t in TARGET_COLS:
            errs = list(self.per_target_errors[t]) if self.per_target_errors[t] else [0.5]
            state.append(np.mean(errs))

        # Pad to state_dim
        while len(state) < self.state_dim:
            state.append(0.0)

        return torch.FloatTensor(state[:self.state_dim])


class RetrainingAction:
    NO_ACTION = 0
    RETRAIN_MRTFN = 1
    RETRAIN_CNN_LSTM = 2
    RETRAIN_TRANSFORMER = 3
    RETRAIN_ALL = 4
    ADJUST_ENSEMBLE_WEIGHTS = 5
    INCREASE_REGULARIZATION = 6

    NAMES = [
        'no_action',
        'retrain_mrtfn',
        'retrain_cnn_lstm',
        'retrain_transformer',
        'retrain_all',
        'adjust_ensemble_weights',
        'increase_regularization'
    ]

    @classmethod
    def num_actions(cls):
        return len(cls.NAMES)

    @classmethod
    def get_name(cls, action_id):
        return cls.NAMES[action_id]


class DQN(nn.Module):

    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.net(state)

    def get_action(self, state, epsilon=0.0):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if random.random() < epsilon:
            return random.randint(0, RetrainingAction.num_actions() - 1)

        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class RLRetrainingAgent:

    def __init__(self, state_dim=20, device='cpu'):
        self.device = device
        self.state_dim = state_dim
        self.n_actions = RetrainingAction.num_actions()

        # DQN networks
        self.policy_net = DQN(state_dim, self.n_actions).to(device)
        self.target_net = DQN(state_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=RL_LEARNING_RATE)

        self.state = RetrainingState(state_dim)
        self.buffer = ReplayBuffer()

        self.epsilon = EPSILON_START
        self.episode_rewards = []
        self.losses = []

    def select_action(self, deterministic=False):
        state_tensor = self.state.to_tensor().to(self.device)
        eps = 0.0 if deterministic else self.epsilon
        return self.policy_net.get_action(state_tensor, eps)

    def compute_reward(self, action, accuracy_before, accuracy_after, error_before, error_after):

        accuracy_change = accuracy_after - accuracy_before
        error_change = error_before - error_after  # Positive if error decreased

        # Base reward from performance change
        reward = accuracy_change * 10 + error_change * 5

        # Action-specific costs/bonuses
        if action == RetrainingAction.NO_ACTION:
            # Good to do nothing if already performing well
            if accuracy_before > 0.9:
                reward += 0.2
            else:
                reward -= 0.1

        elif action in [RetrainingAction.RETRAIN_MRTFN,
                        RetrainingAction.RETRAIN_CNN_LSTM,
                        RetrainingAction.RETRAIN_TRANSFORMER]:
            # Single model retrain cost
            reward -= 0.05
            if accuracy_change > 0.01:
                reward += 0.3  # Bonus for successful retrain

        elif action == RetrainingAction.RETRAIN_ALL:
            # Higher cost for retraining all
            reward -= 0.15
            if accuracy_change > 0.02:
                reward += 0.5  # Higher bonus for significant improvement

        elif action == RetrainingAction.ADJUST_ENSEMBLE_WEIGHTS:
            reward -= 0.02  # Low cost action

        elif action == RetrainingAction.INCREASE_REGULARIZATION:
            reward -= 0.02

        return reward

    def step(self, action, accuracy_before, accuracy_after, error_before, error_after):

        state = self.state.to_tensor()
        reward = self.compute_reward(action, accuracy_before, accuracy_after,
                                     error_before, error_after)

        # Update time since retrain
        action_to_model = {
            RetrainingAction.RETRAIN_MRTFN: 'mrtfn',
            RetrainingAction.RETRAIN_CNN_LSTM: 'cnn_lstm',
            RetrainingAction.RETRAIN_TRANSFORMER: 'transformer',
            RetrainingAction.RETRAIN_ALL: 'all'
        }
        if action in action_to_model:
            self.state.reset_retrain_time(action_to_model[action])

        next_state = self.state.to_tensor()
        self.buffer.push(state, action, reward, next_state, False)

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        return reward

    def train_step(self, batch_size=32):

        if len(self.buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

        return {'loss': loss.item()}

    def update_target_network(self):
        tau = 0.01
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, path)
        print(f"  RL agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint.get('losses', [])


def load_data_and_models():
    print("Loading data and models...")

    data = np.load(DATA_PATH, allow_pickle=True)

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train'][:2000]),  # Subset for faster retraining
        torch.FloatTensor(data['y_single_train'][:2000]),
        torch.FloatTensor(data['y_multi_train'][:2000])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_single_val']),
        torch.FloatTensor(data['y_multi_val'])
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_features = data['X_train'].shape[2]
    n_targets = len(TARGET_COLS)

    print(f"  Train subset: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Features: {n_features}, Targets: {n_targets}")

    # Create and load models
    models = create_models(n_features, n_targets, PREDICTION_HORIZONS, DEVICE)

    for name in ['mrtfn', 'cnn_lstm', 'transformer']:
        model_path = f"{MODELS_PATH}/{name}.pt"
        if Path(model_path).exists():
            models[name].load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            models[name].eval()
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {model_path} not found")

    ensemble_path = f"{MODELS_PATH}/meta_ensemble.pt"
    if Path(ensemble_path).exists():
        models['ensemble'].load_state_dict(torch.load(ensemble_path, map_location=DEVICE, weights_only=True))
        models['ensemble'].eval()
        print(f"  Loaded ensemble")

    return train_loader, val_loader, models, n_features, n_targets


def evaluate_ensemble(ensemble, val_loader):
    ensemble.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            x, y_single, _ = [b.to(DEVICE) for b in batch]
            output = ensemble(x)
            all_preds.append(output['single_step'].cpu().numpy())
            all_targets.append(y_single.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Overall metrics
    mse = np.mean((preds - targets) ** 2)
    r2 = 1 - np.sum((targets - preds) ** 2) / (np.sum((targets - targets.mean()) ** 2) + 1e-8)

    # Per-target errors
    target_errors = {}
    for i, t in enumerate(TARGET_COLS):
        target_errors[t] = np.mean((preds[:, i] - targets[:, i]) ** 2)

    return r2, mse, target_errors


def evaluate_model(model, val_loader):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in val_loader:
            x, y_single, _ = [b.to(DEVICE) for b in batch]
            output = model(x)
            error = ((output['single_step'] - y_single) ** 2).mean().item()
            errors.append(error)

    return np.mean(errors)


def retrain_model(model, train_loader, epochs=3, lr=0.0001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for batch in train_loader:
            x, y_single, y_multi = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output['single_step'], y_single) + criterion(output['multi_step'], y_multi)
            loss.backward()
            optimizer.step()

    model.eval()


def plot_training_results(agent, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Episode rewards
    ax1 = axes[0]
    ax1.plot(agent.episode_rewards, alpha=0.6)
    if len(agent.episode_rewards) > 10:
        window = min(10, len(agent.episode_rewards))
        smoothed = np.convolve(agent.episode_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(agent.episode_rewards)), smoothed, 'r-', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training loss
    ax2 = axes[1]
    if agent.losses:
        ax2.plot(agent.losses, alpha=0.6)
        if len(agent.losses) > 20:
            window = 20
            smoothed = np.convolve(agent.losses, np.ones(window) / window, mode='valid')
            ax2.plot(range(window - 1, len(agent.losses)), smoothed, 'r-', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('DQN Training Loss')
    ax2.grid(True, alpha=0.3)

    # Epsilon decay
    ax3 = axes[2]
    epsilons = [EPSILON_START * (EPSILON_DECAY ** i) for i in range(N_EPISODES)]
    epsilons = [max(EPSILON_END, e) for e in epsilons]
    ax3.plot(epsilons)
    ax3.axhline(EPSILON_END, color='r', linestyle='--', label=f'Min ε = {EPSILON_END}')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/rl_training_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rl_training_results.png")


def main():
    print(f"Episodes: {N_EPISODES}")

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Load data and models
    train_loader, val_loader, models, n_features, n_targets = load_data_and_models()

    # Create RL agent
    agent = RLRetrainingAgent(state_dim=20, device=DEVICE)

    print("Starting RL Training")

    # Track action distribution
    action_counts = {name: 0 for name in RetrainingAction.NAMES}

    for episode in range(N_EPISODES):
        episode_reward = 0

        # Evaluate current performance
        accuracy_before, error_before, target_errors = evaluate_ensemble(models['ensemble'], val_loader)

        # Calculate model confidences based on individual performance
        confidences = {}
        for name in ['mrtfn', 'cnn_lstm', 'transformer']:
            model_error = evaluate_model(models[name], val_loader)
            confidences[name] = 1.0 / (1.0 + model_error)

        # Simulate data drift (in practice, would be calculated from real data distribution)
        drift = np.random.random() * 0.1 + 0.01 * (episode / N_EPISODES)

        # Update agent state
        agent.state.update(error_before, drift, confidences, accuracy_before, target_errors)

        # Select action
        action = agent.select_action(deterministic=False)
        action_name = RetrainingAction.get_name(action)
        action_counts[action_name] += 1

        # Execute action
        if action == RetrainingAction.RETRAIN_MRTFN:
            retrain_model(models['mrtfn'], train_loader, epochs=2)
        elif action == RetrainingAction.RETRAIN_CNN_LSTM:
            retrain_model(models['cnn_lstm'], train_loader, epochs=2)
        elif action == RetrainingAction.RETRAIN_TRANSFORMER:
            retrain_model(models['transformer'], train_loader, epochs=2)
        elif action == RetrainingAction.RETRAIN_ALL:
            for name in ['mrtfn', 'cnn_lstm', 'transformer']:
                retrain_model(models[name], train_loader, epochs=1)
        # Other actions are "soft" adjustments (simulated)

        # Evaluate after action
        accuracy_after, error_after, _ = evaluate_ensemble(models['ensemble'], val_loader)

        # Update RL agent and get reward
        reward = agent.step(action, accuracy_before, accuracy_after, error_before, error_after)
        episode_reward += reward

        # Train RL agent
        if episode > 5:
            agent.train_step(batch_size=BATCH_SIZE)
            agent.update_target_network()

        agent.episode_rewards.append(episode_reward)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            print(f"Episode {episode + 1:3d}: Action={action_name:25s}, "
                  f"R²={accuracy_after:.4f}, ε={agent.epsilon:.3f}, "
                  f"Reward={reward:+.3f}, Avg(10)={avg_reward:+.3f}")

    # Print action distribution

    print("Action Distribution")

    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / N_EPISODES * 100
        print(f"  {name:25s}: {count:3d} ({pct:5.1f}%)")

    # Save results

    print("Saving Results")

    agent.save(f"{OUTPUT_PATH}/rl_agent.pt")
    plot_training_results(agent, OUTPUT_PATH)

    def to_python(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_python(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        return obj

    # Save training summary
    summary = {
        'episodes': N_EPISODES,
        'final_epsilon': agent.epsilon,
        'total_rewards': agent.episode_rewards,
        'action_distribution': action_counts,
        'final_accuracy': float(accuracy_after),
        'config': {
            'gamma': GAMMA,
            'lr': RL_LEARNING_RATE,
            'epsilon_start': EPSILON_START,
            'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY
        }
    }

    with open(f"{OUTPUT_PATH}/rl_training_summary.json", 'w') as f:
        json.dump(to_python(summary), f, indent=2)

    print("RL TRAINING COMPLETE")


if __name__ == "__main__":
    main()

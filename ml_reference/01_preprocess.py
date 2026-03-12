import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION

PONDS_PATH = "data/Ponds1.csv"
WEATHER_PATH = "data/open-meteo-26_61N83_15E79m.xlsx"
OUTPUT_PATH = "data/processed/"

SEQUENCE_LENGTH = 24  # 24 x 20min = 8 hours
PREDICTION_HORIZONS = [3, 6, 18, 36, 72, 144]
TARGET_COLS = ['ph', 'water_temp', 'turbidity']

# Smoothing window for targets
SMOOTHING_WINDOW = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_and_clean_data(pond_path, weather_path):
    """Load and clean data with outlier removal"""
    print("Loading data...")

    # Load pond data
    df = pd.read_csv(pond_path, low_memory=False)
    df['station'] = df['station'].str.lower().str.strip()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])

    df = df.rename(columns={
        'NITRATE(PPM)': 'nitrate', 'PH': 'ph', 'AMMONIA(mg/l)': 'ammonia',
        'TEMP': 'water_temp', 'TURBIDITY': 'turbidity',
        'MANGANESE(mg/l)': 'manganese'
    })

    # Convert to numeric
    numeric_cols = ['nitrate', 'ph', 'ammonia', 'water_temp', 'turbidity', 'manganese']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove obvious outliers for targets
    for col in TARGET_COLS:
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        before = len(df)
        df = df[(df[col] >= q1) & (df[col] <= q99)]
        print(f"  {col}: removed {before - len(df)} outliers")

    # Load weather
    weather = pd.read_excel(weather_path, skiprows=3)
    weather.columns = ['datetime', 'air_temp', 'wind_speed', 'rain', 'col5', 'col6']
    weather = weather[['datetime', 'air_temp', 'wind_speed', 'rain']].copy()
    weather = weather[weather['datetime'] != 'time']
    weather['datetime'] = pd.to_datetime(weather['datetime'], errors='coerce')
    weather = weather.dropna(subset=['datetime'])

    for col in ['air_temp', 'wind_speed', 'rain']:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')

    # Merge weather
    df['hour'] = df['datetime'].dt.floor('H')
    weather['hour'] = weather['datetime'].dt.floor('H')
    df = df.merge(weather[['hour', 'air_temp', 'wind_speed', 'rain']], on='hour', how='left')

    for col in ['air_temp', 'wind_speed', 'rain']:
        df[col] = df[col].ffill().bfill()

    df = df.sort_values(['station', 'datetime']).reset_index(drop=True)
    print(f"Total records: {len(df)}")

    return df


def smooth_and_engineer_features(df, smooth_window=5):
    print(f"Smoothing targets (window={smooth_window})...")

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 7)
    df['is_daytime'] = ((df['datetime'].dt.hour >= 6) & (df['datetime'].dt.hour < 18)).astype(float)

    # Process per station
    smoothed_dfs = []

    for station in df['station'].unique():
        sdf = df[df['station'] == station].copy().sort_values('datetime')

        for col in TARGET_COLS:
            sdf[col] = sdf[col].rolling(smooth_window, center=True, min_periods=1).mean()

        # Lag features
        for col in TARGET_COLS:
            for lag in [1, 2, 3, 6, 12]:
                sdf[f'{col}_lag{lag}'] = sdf[col].shift(lag)

        # Rolling stats on smoothed data
        for col in TARGET_COLS:
            sdf[f'{col}_roll_mean_6'] = sdf[col].rolling(6, min_periods=1).mean()
            sdf[f'{col}_roll_std_6'] = sdf[col].rolling(6, min_periods=1).std().fillna(0)
            sdf[f'{col}_diff'] = sdf[col].diff().fillna(0)

        # Weather features
        for col in ['air_temp', 'wind_speed', 'rain']:
            if col in sdf.columns:
                sdf[f'{col}_lag1'] = sdf[col].shift(1)
                sdf[f'{col}_lag3'] = sdf[col].shift(3)

        # Temperature difference
        if 'air_temp' in sdf.columns:
            sdf['temp_diff'] = sdf['air_temp'] - sdf['water_temp']

        smoothed_dfs.append(sdf)

    df = pd.concat(smoothed_dfs, ignore_index=True)
    df = df.ffill().bfill()

    # Fill any remaining NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"  Features created: {len(df.columns)}")
    return df


def create_sequences(df, seq_len, horizons, target_cols):
    print(f"Creating sequences (length={seq_len})...")

    exclude = ['datetime', 'station', 'Date', 'Time', 'hour', 'ammonia', 'manganese', 'nitrate']
    feature_cols = [c for c in df.columns if c not in exclude and c not in target_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    print(f"  Using {len(feature_cols)} features")

    all_X, all_y_single, all_y_multi = [], [], []
    max_horizon = max(horizons)

    for station in df['station'].unique():
        sdf = df[df['station'] == station].reset_index(drop=True)

        features = sdf[feature_cols].values.astype(np.float32)
        targets = sdf[target_cols].values.astype(np.float32)

        count = 0
        for i in range(seq_len, len(sdf) - max_horizon):
            X = features[i - seq_len:i]
            y_single = targets[i]
            y_multi = np.array([targets[i + h] for h in horizons])

            all_X.append(X)
            all_y_single.append(y_single)
            all_y_multi.append(y_multi)
            count += 1

        print(f"    {station}: {count} sequences")

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y_single, dtype=np.float32),
            np.array(all_y_multi, dtype=np.float32),
            feature_cols)


def split_and_normalize(X, y_single, y_multi, train_ratio, val_ratio):
    n = len(X)
    idx = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_single_train = y_single[train_idx]
    y_single_val = y_single[val_idx]
    y_single_test = y_single[test_idx]
    y_multi_train = y_multi[train_idx]
    y_multi_val = y_multi[val_idx]
    y_multi_test = y_multi[test_idx]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize with train stats
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_single_mean = y_single_train.mean(axis=0, keepdims=True)
    y_single_std = y_single_train.std(axis=0, keepdims=True) + 1e-8

    y_multi_mean = y_multi_train.mean(axis=(0, 1), keepdims=True)
    y_multi_std = y_multi_train.std(axis=(0, 1), keepdims=True) + 1e-8

    y_single_train_n = (y_single_train - y_single_mean) / y_single_std
    y_single_val_n = (y_single_val - y_single_mean) / y_single_std
    y_single_test_n = (y_single_test - y_single_mean) / y_single_std

    y_multi_train_n = (y_multi_train - y_multi_mean) / y_multi_std
    y_multi_val_n = (y_multi_val - y_multi_mean) / y_multi_std
    y_multi_test_n = (y_multi_test - y_multi_mean) / y_multi_std

    norm_params = {
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'y_single_mean': y_single_mean.tolist(),
        'y_single_std': y_single_std.tolist(),
        'y_multi_mean': y_multi_mean.tolist(),
        'y_multi_std': y_multi_std.tolist(),
        'y_single_mean_raw': y_single_mean.flatten().tolist(),
        'y_single_std_raw': y_single_std.flatten().tolist()
    }

    return (X_train, X_val, X_test,
            y_single_train_n, y_single_val_n, y_single_test_n,
            y_multi_train_n, y_multi_val_n, y_multi_test_n,
            norm_params)


def main():
    np.random.seed(42)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = load_and_clean_data(PONDS_PATH, WEATHER_PATH)

    # Smooth and engineer features
    df = smooth_and_engineer_features(df, SMOOTHING_WINDOW)

    # Create sequences
    X, y_single, y_multi, feature_cols = create_sequences(
        df, SEQUENCE_LENGTH, PREDICTION_HORIZONS, TARGET_COLS
    )

    # Split and normalize
    (X_train, X_val, X_test,
     y_single_train, y_single_val, y_single_test,
     y_multi_train, y_multi_val, y_multi_test,
     norm_params) = split_and_normalize(X, y_single, y_multi, TRAIN_RATIO, VAL_RATIO)

    # Save
    np.savez(
        f'{OUTPUT_PATH}/sequences.npz',
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_single_train=y_single_train, y_single_val=y_single_val, y_single_test=y_single_test,
        y_multi_train=y_multi_train, y_multi_val=y_multi_val, y_multi_test=y_multi_test,
        feature_cols=feature_cols, target_cols=TARGET_COLS
    )

    with open(f'{OUTPUT_PATH}/norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)


if __name__ == "__main__":
    main()

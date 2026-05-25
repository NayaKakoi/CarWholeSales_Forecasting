import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging


def jalankan_lstm_dinamis(df_input, time_steps=12, train_split=0.8):
    # ── 1. Agregasi ke nasional (1 baris per bulan) ──────────────────────────
    df_nasional = (
        df_input.groupby('Tanggal')['Aktual']
        .sum()
        .reset_index()
        .sort_values('Tanggal')
        .reset_index(drop=True)
    )

    n_total = len(df_nasional)
    logging.info(f"[LSTM] Total bulan data nasional: {n_total}")

    if n_total < time_steps + 15:
        logging.warning(f"[LSTM] Data tidak cukup ({n_total} bulan). Butuh minimal {time_steps + 15}.")
        return None, 0.0, 0.0, 0.0, None

    # ── 2. Scaling ──────────────────────────────────────────────────────────
    data = df_nasional['Aktual'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    # ── 3. Sliding window features ─────────────────────────────────────────
    X, y = [], []
    for i in range(n_total - time_steps):
        X.append(scaled[i:i + time_steps, 0])
        y.append(scaled[i + time_steps, 0])
    X, y = np.array(X), np.array(y)

    # ── 4. Split: pastikan test set minimal 12 titik (1 tahun) ─────────────
    min_test = max(12, int(len(X) * (1 - train_split)))
    split_idx = len(X) - min_test
    if split_idx < time_steps:
        logging.warning(f"[LSTM] Train set terlalu kecil setelah split ({split_idx} sampel).")
        return None, 0.0, 0.0, 0.0, None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logging.info(f"[LSTM] Train: {len(X_train)} | Test: {len(X_test)} sampel")

    # Reshape untuk LSTM: (samples, timesteps, features)
    X_train_3d = X_train.reshape(-1, time_steps, 1)
    X_test_3d  = X_test.reshape(-1,  time_steps, 1)
    X_all_3d   = X.reshape(-1,       time_steps, 1)

    # ── 5. Model ────────────────────────────────────────────────────────────
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='huber')  # Huber robust vs outlier COVID

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-5, verbose=0),
    ]

    model.fit(
        X_train_3d, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_test_3d, y_test),
        callbacks=callbacks,
        verbose=0,
    )

    # ── 6. Evaluasi ─────────────────────────────────────────────────────────
    pred_test = scaler.inverse_transform(
        model.predict(X_test_3d, verbose=0)
    ).flatten()
    pred_all = scaler.inverse_transform(
        model.predict(X_all_3d, verbose=0)
    ).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_test_actual, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_test))
    r2   = r2_score(y_test_actual, pred_test)

    logging.info(f"[LSTM] MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    return model, mae, rmse, r2, pred_all

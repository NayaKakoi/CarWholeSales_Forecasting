import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

def jalankan_xgboost_dinamis(df_input, time_steps=36, train_split=0.8):
    # ── 1. Agregasi ke nasional (1 baris per bulan) ──────────────────────────
    df_nasional = (
        df_input.groupby('Tanggal')['Aktual']
        .sum()
        .reset_index()
        .sort_values('Tanggal')
        .reset_index(drop=True)
    )

    n_total = len(df_nasional)
    logging.info(f"[XGBoost] Total bulan data nasional: {n_total}")

    # Menurunkan batas aman minimum agar bisa diproses untuk skenario 2 tahun
    if n_total <= time_steps:
        logging.warning(f"[XGBoost] Data terlalu sedikit ({n_total} bulan). Butuh > {time_steps}.")
        return None, 0.0, 0.0, 0.0, None

    data = df_nasional['Aktual'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_total - time_steps):
        X.append(scaled[i:i + time_steps, 0])
        y.append(scaled[i + time_steps, 0])
    X, y = np.array(X), np.array(y)

    # Turunkan batas minimal test dari 12 menjadi 3 untuk mengakomodasi data pendek
    min_test = max(3, int(len(X) * (1 - train_split)))
    split_idx = len(X) - min_test
    if split_idx < 1:
        logging.warning(f"[XGBoost] Train set terlalu kecil setelah split ({split_idx} sampel).")
        return None, 0.0, 0.0, 0.0, None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logging.info(f"[XGBoost] Train: {len(X_train)} | Test: {len(X_test)} sampel")

    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    pred_test_scaled = model_xgb.predict(X_test)
    pred_all_scaled  = model_xgb.predict(X)

    pred_test = scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
    pred_all  = scaler.inverse_transform(pred_all_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_test_actual, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_test))
    r2   = r2_score(y_test_actual, pred_test)

    logging.info(f"[XGBoost] MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    return model_xgb, mae, rmse, r2, pred_all
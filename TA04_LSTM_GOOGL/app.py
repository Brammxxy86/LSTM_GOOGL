"""
TA-04 | Peramalan Harga Penutupan Saham Alphabet (GOOGL) dengan LSTM
Mahasiswa: Bayu Muhammad Ramadhan | NIM: 301240008 | Kelas: 5A
Aplikasi berjalan lokal melalui Flask. Data diambil dari Yahoo Finance saat pertama dijalankan.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"
for folder in (DATA_DIR, ARTIFACT_DIR, STATIC_DIR):
    folder.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "GOOGL_stock_data.csv"
MODEL_PATH = ARTIFACT_DIR / "lstm_googl.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
TIMESTEPS = 60
EPOCHS = int(os.getenv("EPOCHS", "20"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "ta04-lstm-local-key")
_cached_results = None

def load_stock_data(refresh=False):
    """Unduh data harian GOOGL dari Yahoo Finance, lalu simpan sebagai CSV lokal."""
    if refresh or not CSV_PATH.exists():
        data = yf.download("GOOGL", period="5y", interval="1d",
                           auto_adjust=True, progress=False)
        if data.empty:
            raise RuntimeError("Data Yahoo Finance kosong. Periksa koneksi internet.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data.columns = [str(c).replace(" ", "_") for c in data.columns]
        date_col = "Date" if "Date" in data.columns else data.columns[0]
        close_col = "Close"
        data = data[[date_col, close_col]].rename(
            columns={date_col: "Date", close_col: "Close"})
        data.to_csv(CSV_PATH, index=False)
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).drop_duplicates("Date")
    df = df.sort_values("Date").set_index("Date")
    if len(df) <= TIMESTEPS + 30:
        raise RuntimeError("Jumlah data tidak cukup untuk pelatihan LSTM.")
    return df

def make_sequences(values, timesteps=TIMESTEPS):
    X, y = [], []
    for i in range(timesteps, len(values)):
        X.append(values[i-timesteps:i, 0])
        y.append(values[i, 0])
    X = np.asarray(X, dtype=np.float32).reshape(-1, timesteps, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y

def train_model(df):
    values = df[["Close"]].values.astype("float32")
    split_point = int(len(values) * 0.8)
    # Fit scaler hanya pada data latih untuk menghindari data leakage.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:split_point])
    scaled = scaler.transform(values)
    X, y = make_sequences(scaled)
    train_size = max(1, split_point - TIMESTEPS)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(TIMESTEPS, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    stop = EarlyStopping(monitor="val_loss", patience=4,
                         restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32,
              validation_split=0.1, callbacks=[stop], verbose=0,
              shuffle=False)
    pred_scaled = model.predict(X_test, verbose=0)
    actual_scaled = y_test.reshape(-1, 1)
    predictions = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(actual_scaled).ravel()
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    result_dates = df.index[TIMESTEPS + train_size:]
    return model, scaler, rmse, mae, actual, predictions, result_dates

def create_charts(df, actual=None, predictions=None, result_dates=None):
    hist_path = STATIC_DIR / "historical.png"
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Close"], label="Harga penutupan")
    plt.title("Riwayat Harga Penutupan Alphabet (GOOGL)")
    plt.xlabel("Tanggal"); plt.ylabel("Harga (USD)")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(hist_path, dpi=140); plt.close()

    pred_path = STATIC_DIR / "prediction.png"
    plt.figure(figsize=(10, 4))
    if actual is not None and predictions is not None:
        plt.plot(result_dates, actual, label="Aktual")
        plt.plot(result_dates, predictions, label="Prediksi LSTM")
    plt.title("Perbandingan Aktual dan Prediksi pada Data Uji")
    plt.xlabel("Tanggal"); plt.ylabel("Harga (USD)")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(pred_path, dpi=140); plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        global _cached_results
        refresh = request.method == "POST" and request.form.get("action") == "refresh"
        df = load_stock_data(refresh=refresh)
        if _cached_results is None or refresh:
            _cached_results = train_model(df)
        model, scaler, rmse, mae, actual, predictions, result_dates = _cached_results
        create_charts(df, actual, predictions, result_dates)
        next_input = scaler.transform(df[["Close"]].values)[-TIMESTEPS:]
        next_scaled = model.predict(next_input.reshape(1, TIMESTEPS, 1), verbose=0)
        next_price = float(scaler.inverse_transform(next_scaled)[0, 0])
        last_price = float(df["Close"].iloc[-1])
        change = next_price - last_price
        return render_template(
            "index.html", rows=len(df), start=df.index.min().strftime("%d %b %Y"),
            end=df.index.max().strftime("%d %b %Y"), last_price=last_price,
            next_price=next_price, change=change, rmse=rmse, mae=mae,
            historical="historical.png", prediction="prediction.png",
            currency="USD")
    except Exception as exc:
        return render_template("error.html", error=str(exc)), 500

@app.route("/health")
def health():
    return {"status": "ok", "application": "TA-04 LSTM GOOGL"}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)

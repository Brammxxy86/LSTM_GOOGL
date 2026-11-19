# TA-04 — Forecasting Harga Saham GOOGL dengan LSTM

**Mahasiswa:** Bayu Muhammad Ramadhan  
**NIM:** 301240008  
**Kelas:** 5A

Aplikasi web Flask untuk memvisualisasikan riwayat harga penutupan Alphabet Inc. (GOOGL), melatih model LSTM, mengevaluasi prediksi pada data uji, dan memperkirakan timestep berikutnya.

## Dataset
Data historis harian GOOGL diambil saat aplikasi berjalan menggunakan `yfinance` (Yahoo Finance), dengan periode lima tahun terakhir dan penyesuaian harga otomatis. Data disimpan lokal di `data/GOOGL_stock_data.csv` setelah berhasil diunduh.

Sumber: https://finance.yahoo.com/quote/GOOGL/history/  
Alternatif dataset publik untuk pemeriksaan struktur kolom: https://github.com/Maria-mbugua/GOOGLE-Stock-Historical-Prices-EDA

## Menjalankan di VS Code (Windows)
1. Ekstrak ZIP, lalu buka folder `TA04_LSTM_GOOGL` di VS Code.
2. Pastikan Python 3.11 64-bit terpasang.
3. Buka Terminal VS Code dan jalankan:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

4. Buka browser: http://127.0.0.1:5000
5. Proses pertama mengunduh data dan melatih model. Proses ini dapat memerlukan beberapa menit. Model dan scaler disimpan pada folder `artifacts/`.

## Alur model
- Data `Date` dikonversi ke datetime dan dijadikan index.
- Target adalah `Close` (harga penutupan).
- Data latih dan uji dipisah secara kronologis (80:20).
- `MinMaxScaler` di-fit hanya pada bagian latih untuk menghindari data leakage.
- Windowing menggunakan 60 timestep untuk memprediksi satu timestep berikutnya.
- Input LSTM berbentuk tensor 3D `(samples, timesteps, features)`.
- Arsitektur: LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1).
- Optimizer Adam, loss Mean Squared Error, EarlyStopping.
- Evaluasi menggunakan RMSE dan MAE pada data uji setelah inverse transform.

## Struktur
```text
TA04_LSTM_GOOGL/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── data/                 # CSV dibuat otomatis setelah pengunduhan
├── artifacts/            # model .keras dan scaler .pkl dibuat saat training
├── static/               # grafik dibuat otomatis
└── templates/
    ├── index.html
    └── error.html
```

## GitHub
Buat repository kosong di GitHub, lalu dari terminal pada folder proyek:

```bash
git init
git add .
git commit -m "TA-04 LSTM GOOGL forecasting app"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPOSITORY.git
git push -u origin main
```

Ganti URL remote dengan repository milik sendiri. Jangan mengunggah `.venv` atau file hasil training yang besar.

## Deployment Railway (konfigurasi disiapkan, belum dilakukan)
Repository disiapkan dengan `Procfile` dan `runtime.txt` agar dapat dihubungkan ke Railway. Untuk deployment nyata, hubungkan repository GitHub ke Railway, atur environment variable `EPOCHS` (opsional), lalu deploy. **Pada pengumpulan ini aplikasi hanya dijalankan secara lokal; belum ada URL Railway aktif.**

## Catatan
Hasil prediksi harga saham bukan nasihat atau rekomendasi investasi. Model univariat ini hanya menggunakan riwayat `Close` dan tidak memasukkan faktor eksternal.

# Modul Pembelajaran & Panduan Instalasi
**Aplikasi Deteksi Visual AI (Quality Control)**

Modul ini disusun untuk memberikan panduan secara detail, lengkap, dan mudah dipelajari mengenai cara instalasi, konfigurasi, dan penggunaan aplikasi deteksi visual AI.

---

## 🏗️ 1. Arsitektur Sistem
Aplikasi ini dibangun menggunakan teknologi mutakhir di bidang kecerdasan buatan dan pengembangan web:
* **Flask**: Framework web berbasis Python yang ringan dan cepat untuk membangun antarmuka pengguna (backend) dan API.
* **YOLO (You Only Look Once)**: Arsitektur *Deep Learning* tingkat lanjut yang dirancang khusus untuk mendeteksi objek secara *real-time* dengan tingkat akurasi dan kecepatan yang sangat tinggi.
* **OpenCV**: *Library Computer Vision* utama yang difungsikan untuk memproses aliran video dari kamera, melakukan filtering (misal: ruang warna HSV), manipulasi gambar, dan analisis kontur.
* **NumPy**: *Library* komputasi numerik di Python yang bekerja sangat efisien untuk memanipulasi matriks piksel gambar hasil tangkapan OpenCV.
* **Deep Learning**: Fondasi kecerdasan buatan yang memungkinkan model komputasi belajar dari representasi data gambar berdimensi tinggi untuk membedakan produk berkualitas baik ("OK") dan produk cacat ("NG").

---

## ⚙️ 2. Persyaratan Sistem
Sebelum melakukan instalasi, pastikan perangkat komputer/laptop Anda memenuhi spesifikasi minimum berikut:
* **Sistem Operasi**: Windows 10/11 (Direkomendasikan), macOS, atau Linux.
* **Perangkat Keras**: Memiliki Webcam/Kamera Eksternal yang berfungsi dengan baik.
* **Perangkat Lunak**: 
  * Python (versi 3.8 - 3.11 direkomendasikan).
  * Web Browser modern (Google Chrome, Firefox, atau Microsoft Edge).

---

## 🚀 3. Panduan Instalasi (Langkah demi Langkah)

Ikuti instruksi berikut secara berurutan untuk memasang aplikasi di komputer Anda:

### Langkah 1: Buka Direktori Proyek
Buka aplikasi **Command Prompt (CMD)**, **PowerShell**, atau Terminal, lalu masuk ke folder utama tempat kode aplikasi disimpan.
```bash
cd c:\laragon\www\deteksi
```

### Langkah 2: Buat & Aktifkan Virtual Environment
*Virtual environment* sangat penting agar library (kumpulan modul Python) untuk aplikasi ini terisolasi dan tidak merusak aplikasi lain di komputer Anda.
1. Buat environment baru:
   ```bash
   python -m venv .venv
   ```
2. Aktifkan environment (Tanda berhasil: muncul tulisan `(.venv)` di awal CMD Anda):
   * **Pengguna Windows**:
     ```bash
     .\.venv\Scripts\activate
     ```
   * **Pengguna Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

### Langkah 3: Instalasi Library (Dependensi)
Setelah virtual environment aktif, instal semua arsitektur dan library yang dibutuhkan (Flask, OpenCV, Numpy, dll):
```bash
pip install -r requirements.txt
```
*(Catatan: Pastikan komputer Anda terhubung ke internet karena proses ini akan mengunduh paket secara otomatis)*

### Langkah 4: Menjalankan Aplikasi
Setelah instalasi selesai, jalankan mesin server lokal dengan perintah:
```bash
python app.py
```
*(Atau Anda bisa langsung mengklik ganda file `run.bat` di dalam folder jika menggunakan Windows).*

### Langkah 5: Mengakses Aplikasi
Buka Web Browser favorit Anda dan ketikkan alamat berikut di bilah URL:
👉 **http://127.0.0.1:5000** atau **http://localhost:5000**

---

## 🔐 4. Hak Akses Pengguna (Role & Login)

Sistem ini didesain menggunakan metode *Role-Based Access Control*. Sistem membatasi fitur berdasarkan wewenang pekerjaan. Gunakan kredensial di bawah ini untuk mengakses sistem sesuai dengan divisi Anda:

| Peran (Role) | Username | Password | Deskripsi Akses |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` | **Akses Penuh**. Mengelola Pengguna, pengaturan Model AI, melihat Dashboard, Evaluasi, Laporan, dan Analisa Hasil. |
| **Quality Control** | `qc` | `qc123` | **Akses Analitik**. Berfokus pada laporan produksi, memantau metrik Dashboard, serta menarik (eksport) rekam jejak Analisa Hasil untuk kualitas mutu. |
| **Operator** | `operator` | `operator123` | **Akses Mesin**. Akses dibatasi khusus pada layar Inspeksi Utama (Kamera) untuk memonitor lini perakitan secara *live*. |

---

## 💡 5. Tips Pemecahan Masalah (Troubleshooting) Singkat
* **Gagal Menjalankan Kamera**: Pastikan webcam tidak sedang digunakan oleh aplikasi lain seperti Zoom atau Google Meet.
* **"Address already in use" Error**: Ini berarti ada aplikasi Flask lain yang sedang berjalan. Tutup paksa Terminal Anda lalu buka kembali, atau *restart* komputer Anda.
* **Error "ModuleNotFoundError"**: Pastikan Anda sudah mengaktifkan *Virtual Environment* (Langkah 2) sebelum mengetikkan perintah `python app.py`.

# 🧠 Waste Management Indonesia - Smart Trash Detection System

Aplikasi berbasis citra digital untuk mendeteksi dan memantau keberadaan sampah di kawasan wisata secara otomatis menggunakan teknologi pemrosesan citra dan machine learning.

## 📌 Deskripsi

Proyek ini bertujuan untuk membantu pengelolaan sampah di sektor pariwisata Indonesia melalui deteksi visual menggunakan kamera (CCTV/drone) dan klasifikasi area kotor/bersih secara real-time. Sistem ini juga dilengkapi dashboard monitoring dan notifikasi otomatis ke petugas kebersihan.

## 🎯 Fitur Utama

- 🔍 Deteksi sampah otomatis (kamera/webcam/drone)
- 🧼 Segmentasi area bersih vs kotor
- 📊 Dashboard monitoring dengan peta dan snapshot
- 🚨 Notifikasi otomatis via Telegram/API
- 📈 Log data dan analisis tren harian/mingguan

## 🧪 Teknologi yang Digunakan

| Komponen           | Teknologi                          |
|--------------------|------------------------------------|
| Bahasa             | Python                             |
| Pemrosesan Citra   | OpenCV                             |
| Deep Learning      | YOLOv5 / TensorFlow + MobileNet    |
| Dashboard          | Streamlit / Flask                  |
| Notifikasi         | Telegram Bot / REST API            |
| Perangkat          | Webcam, CCTV, Drone (optional)     |

## 🚀 Instalasi

```bash
# Clone repo
git clone https://github.com/username/waste-management-indonesia.git
cd waste-management-indonesia

# (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate  # atau .\\venv\\Scripts\\activate di Windows

# Install dependencies
pip install -r requirements.txt
```
##🖼️ Struktur Folder
waste-management-indonesia/
├── dataset/
├── models/
├── src/
│   ├── detect.py
│   ├── segment.py
│   └── notify.py
├── dashboard/
│   └── app.py
├── requirements.txt
└── README.md

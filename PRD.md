1. Informasi Proyek & Ringkasan Eksekutif
Nama Proyek: ShelfSense Edge

Sektor: Sustainability, Retail Tech, & Food Waste Management

Visi Produk: Mengubah manajemen inventaris produk segar dari berbasis tanggal statis (FIFO/expiration date) menjadi berbasis kondisi biologis aktual secara real-time menggunakan Edge Computer Vision dan Dynamic Pricing otomatis.

Target Demo Utama (The "3-Banana Test"): Mendeteksi 3 buah pisang dengan tingkat kematangan berbeda (Hijau/Segar, Kuning/Matang, Bercak Coklat/Terlalu Matang) secara bersamaan melalui webcam, menampilkan bounding box, indeks kematangan, estimasi sisa umur simpan (shelf-life), serta perubahan harga otomatis pada simulasi Electronic Shelf Label (ESL) secara live.

2. Arsitektur Teknis & Tech Stack
Frontend: Next.js (App Router, TypeScript), Tailwind CSS, Shadcn UI, Lucide Icons, Recharts (untuk kurva pembusukan dan metrik dampak karbon), Framer Motion (animasi perubahan harga pada ESL).

Backend: Python 3.10+, FastAPI, Uvicorn, WebSockets (untuk streaming frame video dan telemetri JSON real-time).

AI & Computer Vision Pipeline:

Object Detection: Ultralytics YOLOv8 (yolov8n.pt atau yolov8s.pt) berjalan di atas PyTorch dengan akselerasi CUDA (NVIDIA RTX 3060). Fokus filter pada kelas COCO relevan: banana (class ID 46), apple (class ID 47), dan orange (class ID 49).

Visual Decay & Ripeness Indexing (OpenCV): Melakukan cropping pada tiap bounding box objek yang terdeteksi, mengonversi ruang warna BGR ke HSV, dan menghitung persentase piksel untuk tiga spektrum utama:

Hijau (Unripe/Fresh): Hue berkisar 35–85.

Kuning/Cerah (Ripe/Prime): Hue berkisar 18–34.

Coklat/Hitam (Overripe/Decay Spots): Hue berkisar 0–18 dengan Value (kecerahan) rendah atau analisis kontur bintik gelap.

Spoilage Time-Series Predictor (LSTM + Environmental Modifier): Model berbasis PyTorch LSTM (dengan pre-trained synthetic weights atau hybrid mathematical decay curve untuk stabilitas demo) yang mengambil input [Ripeness_Index, Spot_Ratio, Room_Temp, Humidity] dan mengeluarkan output estimasi sisa jam (Remaining Shelf-Life in Hours) serta proyeksi kurva penurunan kualitas 72 jam ke depan.


3. Spesifikasi Fitur Utama (Functional Requirements)
ID : FR-01
Modul Fitur : Multi-Object Live CV Engine
Status Eksekusi : LIVE & REAL
Deskripsi Teknis & Perilaku Sistem : Membaca input webcam via OpenCV, mendeteksi beberapa buah sekaligus dengan YOLOv8, memberikan bounding box berwarna sesuai status (Hijau, Kuning, Merah), dan menampilkan ID unik tiap item di layar.

ID : FR-02
Modul Fitur : HSV Ripeness & Spot Analyzer
Status Eksekusi : LIVE & REAL
Deskripsi Teknis & Perilaku Sistem : Melakukan cropping pada tiap bounding box objek, mengonversi ruang warna BGR ke HSV, serta menghitung persentase piksel untuk spektrum hijau (unripe), kuning (ripe), dan bintik coklat/hitam (decay spots) guna menentukan skor kematangan secara akurat.

ID : FR-03
Modul Fitur : Spoilage & Shelf-Life Predictor
Status Eksekusi : LIVE & REAL
Deskripsi Teknis & Perilaku Sistem : Mengonversi skor visual dan parameter lingkungan (suhu ruangan & kelembaban) menggunakan model berbasis PyTorch LSTM dan kurva pembusukan biologis untuk memprediksi sisa umur simpan secara real-time (contoh: "5 Hari", "2 Hari", "8 Jam").

ID : FR-04
Modul Fitur : Automated Dynamic Pricing & ESL Simulator
Status Eksekusi : LIVE & REAL
Deskripsi Teknis & Perilaku Sistem : Mengalkulasi diskon otomatis (0%, 15%, 30%, 50%) berdasarkan sisa umur simpan produk, lalu secara real-time memperbarui kartu Electronic Shelf Label (ESL) digital di layar monitor untuk masing-masing buah yang terdeteksi.

ID : FR-05

Modul Fitur : Calibration & Env Control Panel

Status Eksekusi : LIVE & REAL

Deskripsi Teknis & Perilaku Sistem : Menyediakan panel kontrol interaktif di interface (UI) untuk mengatur simulasi suhu ruangan (15°C – 38°C) dan kalibrasi threshold HSV secara langsung guna menguji laju pembusukan di depan juri


ID : FR-06

Modul Fitur : Consumer "Near-Me" Flash Deal Mockup

Status Eksekusi : MOCKUP / UI

Deskripsi Teknis & Perilaku Sistem : Menampilkan simulasi aplikasi mobile konsumen yang secara otomatis memunculkan pop-up push notification saat buah di rak supermarket mendapati diskon flash sale (30% – 50%).

ID : FR-07

Modul Fitur : ESG & Carbon Impact Analytics

Status Eksekusi : MOCKUP / UIDeskripsi Teknis & Perilaku Sistem : Menyajikan dashboard keberlanjutan bulanan yang merangkum total kilogram makanan yang berhasil diselamatkan, estimasi reduksi emisi karbon, serta pemulihan margin finansial ritel.

4. Logika Kalkulasi Dynamic Pricing (Aturan Bisnis)
Setiap objek buah yang terdeteksi memiliki Harga Dasar (Base Price) (contoh: Rp 25.000/sisir atau Rp 5.000/buah) dan diklasifikasikan secara real-time ke dalam 4 kuadran:

Grade A — Fresh / Unripe (Skor Kerusakan di bawah 15%, Sisa Umur lebih dari 72 Jam):

Status: OPTIMAL STOCK

Diskon: 0% (Harga Normal)

Warna Indikator ESL: Hijau Zamrud

Grade B — Prime Ripe (Skor Kerusakan 15% – 35%, Sisa Umur 36 – 72 Jam):

Status: PRIME CONDITION - PROMOTE

Diskon: 15% OFF

Warna Indikator ESL: Kuning Amber

Grade C — Fast-Moving / Early Decay (Skor Kerusakan 36% – 60%, Sisa Umur 12 – 35 Jam):

Status: FLASH SALE TRIGGERED

Diskon: 30% OFF (Memicu notifikasi ke aplikasi konsumen sekitar)

Warna Indikator ESL: Oranye

Grade D — Critical Spoilage Risk (Skor Kerusakan di atas 60%, Sisa Umur di bawah 12 Jam):

Status: CLEARANCE / LAST CALL

Diskon: 50% OFF

Warna Indikator ESL: Merah Menyala (Berdenyut/Pulse)



5. Desain Antarmuka (UI/UX Layout)
Dashboard dirancang dalam satu layar Command Center (Dark Mode modern ala sistem IoT Enterprise) dengan 4 area utama:
Kiri Atas (Utama - 60% Lebar): Live Edge AI Camera Feed lengkap dengan overlay HUD (FPS counter, GPU CUDA status, bounding boxes berlapis informasi kematangan & harga). Di bawahnya terdapat Environmental & Calibration Bar (Slider Suhu Ruangan & Kelembaban untuk mendemonstrasikan bagaimana kenaikan suhu mempercepat prediksi pembusukan secara live di depan juri).

Kanan Atas (40% Lebar): Live Electronic Shelf Labels (ESL) Grid. Menampilkan kartu-kartu label harga elektronik digital (gaya layar E-Ink dengan aksen warna LED) yang tersinkronisasi 1-ke-1 dengan buah yang sedang tertangkap kamera.

Kiri Bawah: LSTM Spoilage Decay Curve. Grafik real-time yang memplot posisi setiap buah saat ini pada kurva penurunan kualitas terhadap waktu dan titik pemicu diskon (price drop thresholds).

Kanan Bawah: Tab Switcher (Mockup Ekosistem):Tab 1: Consumer App Simulator (Mockup layar HP dengan notifikasi Flash Deal real-time saat pisang bercak coklat terdeteksi).Tab 2: Sustainability & ESG Impact (Kartu metrik co2 yang diselamatkan, food waste prevented, dan proyeksi revenue recovery bulanan).
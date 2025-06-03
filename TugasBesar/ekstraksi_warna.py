import cv2
import numpy as np
import os 
import csv

def ekstraksi_fitur_warna(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None

    image = cv2.resize(image, (200, 200))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    mean_rgb = cv2.mean(image)[:3] 

    fitur = np.hstack((hist, mean_rgb))

    return fitur

def proses_folder_dataset(folder_dataset, output_csv):
    data = []
    label_list = []

    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue

        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_warna(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)

    with open(output_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for fitur, label in zip(data, label_list):
            writer.writerow(np.append(fitur, label))

    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_csv}")

def hitung_jarak(fitur1, fitur2):
    """
    Menghitung jarak Euclidean antara dua fitur
    """
    return np.sqrt(np.sum((fitur1 - fitur2) ** 2))

def klasifikasi_gambar(fitur_test, fitur_training, label_training):
    """
    Klasifikasi menggunakan metode nearest neighbor
    """
    jarak_min = float('inf')
    label_prediksi = None
    
    # Cari jarak terdekat
    for fitur_train, label in zip(fitur_training, label_training):
        jarak = hitung_jarak(fitur_test, fitur_train)
        if jarak < jarak_min:
            jarak_min = jarak
            label_prediksi = label
            
    return label_prediksi

def load_dataset_from_csv(csv_path):
    """
    Membaca dataset dari file CSV
    """
    features = []
    labels = []
    
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Label ada di kolom terakhir
            features.append([float(x) for x in row[:-1]])
            labels.append(row[-1])
    
    return np.array(features), np.array(labels)

def predict_image(image_path, features_training, labels_training):
    """
    Prediksi kelas untuk gambar baru
    """
    fitur = ekstraksi_fitur_warna(image_path)
    if fitur is not None:
        label_prediksi = klasifikasi_gambar(fitur, features_training, labels_training)
        return label_prediksi
    return None

if __name__ == "__main__":
    # Gunakan absolute path untuk dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset")
    output_csv_path = os.path.join(current_dir, "hasil_ekstraksi/fitur_warna.csv")

    # Pastikan direktori hasil ekstraksi ada
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Proses dataset
    print("[INFO] Memulai ekstraksi fitur dari dataset...")
    proses_folder_dataset(dataset_path, output_csv_path)
    
    # Load dataset untuk klasifikasi
    print("\n[INFO] Loading dataset...")
    features_training, labels_training = load_dataset_from_csv(output_csv_path)    # Test semua gambar dari dataset
    test_images = [
        os.path.join(dataset_path, "organik", "pisang.jpeg"),
        os.path.join(dataset_path, "organik", "kertas.png"),  # kertas dikategorikan sebagai sampah organik
        os.path.join(dataset_path, "anorganik", "plastik.jpg")
    ]
        
    print("\n[INFO] Hasil Klasifikasi:")
    print("-" * 40)
    for test_image in test_images:
        if os.path.exists(test_image):
            hasil_prediksi = predict_image(test_image, features_training, labels_training)
            if hasil_prediksi:
                print(f"{os.path.basename(test_image)} = {hasil_prediksi}")

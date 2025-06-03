import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

def ekstraksi_fitur_tekstur(image_path):
    """
    Ekstraksi fitur tekstur menggunakan GLCM (Gray Level Co-occurrence Matrix)
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None

    # Resize gambar
    image = cv2.resize(image, (200, 200))
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Hitung GLCM
    distances = [1]  # jarak antar pixel
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # sudut 0°, 45°, 90°, 135°
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # Ekstrak fitur dari GLCM
    fitur = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in properties:
        fitur.extend(graycoprops(glcm, prop).flatten())
    
    return np.array(fitur)

def proses_folder_dataset(folder_dataset, output_file):
    """
    Proses semua gambar dalam folder dataset
    """
    data = []
    label_list = []

    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue

        print(f"[INFO] Memproses folder {label}...")
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_tekstur(path_file)
            
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)

    # Simpan hasil ke file numpy
    np.savez(output_file, 
             fitur=np.array(data), 
             label=np.array(label_list))
    
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_file}")
    return np.array(data), np.array(label_list)

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
    
    for fitur_train, label in zip(fitur_training, label_training):
        jarak = hitung_jarak(fitur_test, fitur_train)
        if jarak < jarak_min:
            jarak_min = jarak
            label_prediksi = label
            
    return label_prediksi, jarak_min

def predict_image(image_path, fitur_training, label_training):
    """
    Prediksi kelas untuk gambar baru
    """
    fitur = ekstraksi_fitur_tekstur(image_path)
    if fitur is not None:
        label_prediksi, jarak = klasifikasi_gambar(fitur, fitur_training, label_training)
        return label_prediksi, jarak
    return None, None

if __name__ == "__main__":
    dataset_path = "dataset"
    output_file = "hasil_ekstraksi/fitur_tekstur.npz"
    
    # Buat direktori hasil jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Proses dataset
    print("[INFO] Memulai ekstraksi fitur dari dataset...")
    fitur_training, label_training = proses_folder_dataset(dataset_path, output_file)
      # Test semua gambar dari dataset
    test_images = [
        os.path.join("dataset", "organik", "pisang.jpeg"),
        os.path.join("dataset", "anorganik", "kertas.png"),
        os.path.join("dataset", "anorganik", "plastik.jpg")
    ]
        
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n[INFO] Testing klasifikasi untuk {os.path.basename(test_image)}...")
            hasil_prediksi, jarak = predict_image(test_image, fitur_training, label_training)
            if hasil_prediksi:
                print(f"[HASIL] Gambar {os.path.basename(test_image)} = {hasil_prediksi}")
                print(f"[INFO] Jarak ke kelas terdekat: {jarak:.4f}")

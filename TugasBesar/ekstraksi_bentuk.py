import cv2
import numpy as np
import os
import csv

def ekstraksi_fitur_bentuk(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in kontur:
        luas = cv2.contourArea(cnt)
        if luas < 100:
            continue
        keliling = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        rasio_aspek = w / h
        approx = cv2.approxPolyDP(cnt, 0.04 * keliling, True)
        sisi = len(approx)
        return [luas, rasio_aspek, sisi]
    return None

def proses_folder_dataset_bentuk(folder_dataset, output_csv):
    data = []
    label_list = []
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_bentuk(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)
    with open(output_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for fitur, label in zip(data, label_list):
            writer.writerow(np.append(fitur, label))
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_csv}")

def hitung_jarak(fitur1, fitur2):
    return np.sqrt(np.sum((np.array(fitur1) - np.array(fitur2)) ** 2))

def knn_predict(fitur_test, fitur_training, label_training, k=3):
    jarak = []
    for fitur_train, label in zip(fitur_training, label_training):
        d = hitung_jarak(fitur_test, fitur_train)
        jarak.append((d, label))
    jarak.sort(key=lambda x: x[0])
    k_terdekat = jarak[:k]
    label_count = {}
    for _, label in k_terdekat:
        label_count[label] = label_count.get(label, 0) + 1
    return max(label_count, key=label_count.get)

def load_dataset_from_csv(csv_path):
    features = []
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features.append([float(x) for x in row[:-1]])
            labels.append(row[-1])
    return np.array(features), np.array(labels)

def predict_image(image_path, features_training, labels_training, k=3):
    fitur = ekstraksi_fitur_bentuk(image_path)
    if fitur is not None:
        label_prediksi = knn_predict(fitur, features_training, labels_training, k)
        return label_prediksi
    return None

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset")
    output_csv_path = os.path.join(base_dir, "hasil_ekstraksi", "fitur_bentuk.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    print("[INFO] Memulai ekstraksi fitur bentuk dari dataset...")
    proses_folder_dataset_bentuk(dataset_path, output_csv_path)
    print("\n[INFO] Loading dataset...")
    features_training, labels_training = load_dataset_from_csv(output_csv_path)
    test_images = [
        os.path.join(dataset_path, "organik", "pisang.jpeg"),
        os.path.join(dataset_path, "anorganik", "plastik.jpg"),
        os.path.join(dataset_path, "organik", "kertas.png")
    ]
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n[INFO] Testing klasifikasi untuk {os.path.basename(test_image)}...")
            hasil_prediksi = predict_image(test_image, features_training, labels_training, k=3)
            if hasil_prediksi:
                print(f"[HASIL] Gambar {os.path.basename(test_image)} = {hasil_prediksi}")

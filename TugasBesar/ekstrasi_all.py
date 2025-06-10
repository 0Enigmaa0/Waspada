import cv2
import numpy as np
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from klasifikasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image
from skimage.feature import graycomatrix, graycoprops

# --- Ekstraksi Fitur Bentuk ---
def ekstraksi_fitur_bentuk(image_path):
    image = cv2.imread(image_path)
    if image is None:
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

# --- Ekstraksi Fitur Tekstur ---
def ekstraksi_fitur_tekstur(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    fitur = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in properties:
        fitur.extend(graycoprops(glcm, prop).flatten())
    return np.array(fitur)

# --- Ekstraksi Fitur Warna ---
def ekstraksi_fitur_warna(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (200, 200))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    mean_rgb = cv2.mean(image)[:3]
    fitur = np.hstack((hist, mean_rgb))
    return fitur

# --- Load Dataset ---
def load_dataset(folder_dataset, ekstraksi_fitur_func):
    data = []
    label_list = []
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_func(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)
    return np.array(data), np.array(label_list)

# --- Main ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset")
    labels = ["organik", "anorganik"]

    # Ekstraksi Bentuk
    print("[INFO] Ekstraksi fitur bentuk...")
    X_bentuk, y_bentuk = load_dataset(dataset_path, ekstraksi_fitur_bentuk)
    model_knn_bentuk = klasifikasi_knn(X_bentuk, y_bentuk, k=3)
    model_svm_bentuk = klasifikasi_svm(X_bentuk, y_bentuk, kernel='linear')
    y_pred_knn_bentuk = model_knn_bentuk.predict(X_bentuk)
    y_pred_svm_bentuk = model_svm_bentuk.predict(X_bentuk)

    # Ekstraksi Tekstur
    print("[INFO] Ekstraksi fitur tekstur...")
    X_tekstur, y_tekstur = load_dataset(dataset_path, ekstraksi_fitur_tekstur)
    model_knn_tekstur = klasifikasi_knn(X_tekstur, y_tekstur, k=3)
    model_svm_tekstur = klasifikasi_svm(X_tekstur, y_tekstur, kernel='linear')
    y_pred_knn_tekstur = model_knn_tekstur.predict(X_tekstur)
    y_pred_svm_tekstur = model_svm_tekstur.predict(X_tekstur)

    # Ekstraksi Warna
    print("[INFO] Ekstraksi fitur warna...")
    X_warna, y_warna = load_dataset(dataset_path, ekstraksi_fitur_warna)
    model_knn_warna = klasifikasi_knn(X_warna, y_warna, k=3)
    model_svm_warna = klasifikasi_svm(X_warna, y_warna, kernel='linear')
    y_pred_knn_warna = model_knn_warna.predict(X_warna)
    y_pred_svm_warna = model_svm_warna.predict(X_warna)

    # --- Classification Report & Confusion Matrix ---
    ekstraksi_list = [
        ("Bentuk", y_bentuk, y_pred_knn_bentuk, y_pred_svm_bentuk),
        ("Tekstur", y_tekstur, y_pred_knn_tekstur, y_pred_svm_tekstur),
        ("Warna", y_warna, y_pred_knn_warna, y_pred_svm_warna)
    ]
    for nama, y_true, y_pred_knn, y_pred_svm in ekstraksi_list:
        print(f"\n=== {nama.upper()} ===")
        print("[KNN] Classification Report:")
        print(classification_report(y_true, y_pred_knn))
        print("[SVM] Classification Report:")
        print(classification_report(y_true, y_pred_svm))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred_knn), display_labels=labels).plot(ax=ax[0], colorbar=False)
        ax[0].set_title(f"{nama} - KNN")
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred_svm), display_labels=labels).plot(ax=ax[1], colorbar=False)
        ax[1].set_title(f"{nama} - SVM")
        plt.tight_layout()
        plt.show()

    # --- Ambil 3 Gambar untuk Visualisasi ---
    contoh_gambar = []
    # 1 organik
    folder_organik = os.path.join(dataset_path, "organik")
    for img in os.listdir(folder_organik):
        contoh_gambar.append((os.path.join(folder_organik, img), "organik"))
        break
    # 2 anorganik
    folder_anorganik = os.path.join(dataset_path, "anorganik")
    count = 0
    for img in os.listdir(folder_anorganik):
        contoh_gambar.append((os.path.join(folder_anorganik, img), "anorganik"))
        count += 1
        if count == 2:
            break
    # Tabel hasil akhir
    print("\nTabel Hasil Klasifikasi (KNN & SVM) untuk 3 Gambar Contoh:")
    print("| Gambar | Label | Bentuk-KNN | Bentuk-SVM | Tekstur-KNN | Tekstur-SVM | Warna-KNN | Warna-SVM |")
    print("|--------|-------|------------|------------|-------------|-------------|-----------|-----------|")
    for img_path, label in contoh_gambar:
        fitur_bentuk = ekstraksi_fitur_bentuk(img_path)
        fitur_tekstur = ekstraksi_fitur_tekstur(img_path)
        fitur_warna = ekstraksi_fitur_warna(img_path)
        pred_bentuk_knn = model_knn_bentuk.predict([fitur_bentuk])[0] if fitur_bentuk is not None else "-"
        pred_bentuk_svm = model_svm_bentuk.predict([fitur_bentuk])[0] if fitur_bentuk is not None else "-"
        pred_tekstur_knn = model_knn_tekstur.predict([fitur_tekstur])[0] if fitur_tekstur is not None else "-"
        pred_tekstur_svm = model_svm_tekstur.predict([fitur_tekstur])[0] if fitur_tekstur is not None else "-"
        pred_warna_knn = model_knn_warna.predict([fitur_warna])[0] if fitur_warna is not None else "-"
        pred_warna_svm = model_svm_warna.predict([fitur_warna])[0] if fitur_warna is not None else "-"
        print(f"| {os.path.basename(img_path)} | {label} | {pred_bentuk_knn} | {pred_bentuk_svm} | {pred_tekstur_knn} | {pred_tekstur_svm} | {pred_warna_knn} | {pred_warna_svm} |")
        # Visualisasi hasil ekstraksi
        img = cv2.imread(img_path)
        if img is not None:
            # Bentuk: tampilkan citra asli, grayscale, biner
            abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imshow(f"Bentuk - Asli: {os.path.basename(img_path)}", img)
            cv2.imshow(f"Bentuk - Grayscale: {os.path.basename(img_path)}", abu)
            cv2.imshow(f"Bentuk - Biner: {os.path.basename(img_path)}", biner)
            # Tekstur: tampilkan grayscale dan biner
            _, biner_tekstur = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imshow(f"Tekstur - Grayscale: {os.path.basename(img_path)}", abu)
            cv2.imshow(f"Tekstur - Biner: {os.path.basename(img_path)}", biner_tekstur)
            # Warna: tampilkan asli dan HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # untuk visualisasi
            cv2.imshow(f"Warna - Asli: {os.path.basename(img_path)}", img)
            cv2.imshow(f"Warna - HSV: {os.path.basename(img_path)}", hsv_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Tabel hasil akhir untuk seluruh dataset
    print("\nTabel Hasil Klasifikasi (KNN & SVM) untuk Seluruh Dataset:")
    print("| Gambar | Label | Bentuk-KNN | Bentuk-SVM | Tekstur-KNN | Tekstur-SVM | Warna-KNN | Warna-SVM |")
    print("|--------|-------|------------|------------|-------------|-------------|-----------|-----------|")
    for label in labels:
        folder_path = os.path.join(dataset_path, label)
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, image_name)
                fitur_bentuk = ekstraksi_fitur_bentuk(img_path)
                fitur_tekstur = ekstraksi_fitur_tekstur(img_path)
                fitur_warna = ekstraksi_fitur_warna(img_path)
                pred_bentuk_knn = model_knn_bentuk.predict([fitur_bentuk])[0] if fitur_bentuk is not None else "-"
                pred_bentuk_svm = model_svm_bentuk.predict([fitur_bentuk])[0] if fitur_bentuk is not None else "-"
                pred_tekstur_knn = model_knn_tekstur.predict([fitur_tekstur])[0] if fitur_tekstur is not None else "-"
                pred_tekstur_svm = model_svm_tekstur.predict([fitur_tekstur])[0] if fitur_tekstur is not None else "-"
                pred_warna_knn = model_knn_warna.predict([fitur_warna])[0] if fitur_warna is not None else "-"
                pred_warna_svm = model_svm_warna.predict([fitur_warna])[0] if fitur_warna is not None else "-"
                print(f"| {image_name} | {label} | {pred_bentuk_knn} | {pred_bentuk_svm} | {pred_tekstur_knn} | {pred_tekstur_svm} | {pred_warna_knn} | {pred_warna_svm} |")

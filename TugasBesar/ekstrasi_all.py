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

# --- Visualisasi Ekstraksi ---
def visualisasi_ekstraksi(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    # 1. Visualisasi Ekstraksi Bentuk
    img_bentuk = img.copy()
    abu = cv2.cvtColor(img_bentuk, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar kontur
    img_kontur = img.copy()
    cv2.drawContours(img_kontur, kontur, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli'), plt.axis('off')
    plt.subplot(132), plt.imshow(abu, cmap='gray')
    plt.title('Grayscale'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(img_kontur, cv2.COLOR_BGR2RGB))
    plt.title('Kontur'), plt.axis('off')
    plt.suptitle('Ekstraksi Bentuk')
    plt.show()

    # 2. Visualisasi Ekstraksi Tekstur
    img_resize = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # Hitung properti GLCM
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_props = {}
    for prop in properties:
        glcm_props[prop] = graycoprops(glcm, prop)[0, 0]
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli'), plt.axis('off')
    plt.subplot(132), plt.imshow(gray, cmap='gray')
    plt.title('Grayscale'), plt.axis('off')
    
    # Visualisasi matriks GLCM
    plt.subplot(133), plt.imshow(glcm[:, :, 0, 0], cmap='hot')
    plt.title('GLCM Matrix')
    plt.colorbar(orientation='vertical')
    props_text = '\n'.join([f'{prop}: {val:.2f}' for prop, val in glcm_props.items()])
    plt.figtext(1.0, 0.5, props_text, fontsize=9, ha='left', va='center')
    plt.suptitle('Ekstraksi Tekstur')
    plt.tight_layout()
    plt.show()

    # 3. Visualisasi Ekstraksi Warna
    img_resize = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('RGB'), plt.axis('off')
    
    # Tampilkan komponen HSV
    h, s, v = cv2.split(hsv)
    plt.subplot(132), plt.imshow(h, cmap='hsv')
    plt.title('Hue Channel'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV'), plt.axis('off')
    plt.suptitle('Ekstraksi Warna')
    plt.show()

# --- Load Dataset ---
def load_dataset(folder_dataset, ekstraksi_fitur_func):
    data = []
    label_list = []
    valid_labels = ['organik', 'anorganik']
    
    for label in os.listdir(folder_dataset):
        # Skip non-directory items and the 'hasil' folder
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label) or label == 'hasil' or label not in valid_labels:
            continue
            
        print(f"[INFO] Processing {label} folder...")
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_func(path_file)
            if fitur is not None:
                data.append(fitur)
                # Convert both metal and plastic to anorganik
                if file.startswith(('metal', 'plastic')):
                    label_list.append('anorganik')
                else:
                    label_list.append(label)
    
    return np.array(data), np.array(label_list)

# --- Simpan Visualisasi ---
def simpan_visualisasi(img_path, output_folder, label, index):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)
    base_name = f"{label}_{index}"

    # 1. Visualisasi dan simpan ekstraksi bentuk
    img_bentuk = img.copy()
    abu = cv2.cvtColor(img_bentuk, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_kontur = img.copy()
    cv2.drawContours(img_kontur, kontur, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli'), plt.axis('off')
    plt.subplot(132), plt.imshow(abu, cmap='gray')
    plt.title('Grayscale'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(img_kontur, cv2.COLOR_BGR2RGB))
    plt.title('Kontur'), plt.axis('off')
    plt.suptitle(f'Ekstraksi Bentuk - {base_name}')
    plt.savefig(os.path.join(output_folder, f'{base_name}_bentuk.png'))
    plt.close()

    # 2. Visualisasi dan simpan ekstraksi tekstur
    img_resize = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_props = {}
    for prop in properties:
        glcm_props[prop] = graycoprops(glcm, prop)[0, 0]
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli'), plt.axis('off')
    plt.subplot(132), plt.imshow(gray, cmap='gray')
    plt.title('Grayscale'), plt.axis('off')
    plt.subplot(133), plt.imshow(glcm[:, :, 0, 0], cmap='hot')
    plt.title('GLCM Matrix')
    plt.colorbar(orientation='vertical')
    props_text = '\n'.join([f'{prop}: {val:.2f}' for prop, val in glcm_props.items()])
    plt.figtext(1.0, 0.5, props_text, fontsize=9, ha='left', va='center')
    plt.suptitle(f'Ekstraksi Tekstur - {base_name}')
    plt.savefig(os.path.join(output_folder, f'{base_name}_tekstur.png'))
    plt.close()

    # 3. Visualisasi dan simpan ekstraksi warna
    img_resize = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('RGB'), plt.axis('off')
    plt.subplot(132), plt.imshow(h, cmap='hsv')
    plt.title('Hue Channel'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV'), plt.axis('off')
    plt.suptitle(f'Ekstraksi Warna - {base_name}')
    plt.savefig(os.path.join(output_folder, f'{base_name}_warna.png'))
    plt.close()

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
    y_pred_svm_warna = model_svm_warna.predict(X_warna)    # --- Classification Report & Confusion Matrix ---
    ekstraksi_list = [
        ("Bentuk", y_bentuk, y_pred_knn_bentuk, y_pred_svm_bentuk),
        ("Tekstur", y_tekstur, y_pred_knn_tekstur, y_pred_svm_tekstur),
        ("Warna", y_warna, y_pred_knn_warna, y_pred_svm_warna)
    ]
    
    for nama, y_true, y_pred_knn, y_pred_svm in ekstraksi_list:
        print(f"\n=== {nama.upper()} ===")
        
        # Ensure we're using the correct labels
        unique_labels = np.unique(y_true)
        target_names = ['organik', 'anorganik']
        
        print("[KNN] Classification Report:")
        print(classification_report(y_true, y_pred_knn, 
                                 target_names=target_names,
                                 labels=['organik', 'anorganik']))
        print("[SVM] Classification Report:")
        print(classification_report(y_true, y_pred_svm, 
                                 target_names=target_names,
                                 labels=['organik', 'anorganik']))
        
        # Plot confusion matrix
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        cm_knn = confusion_matrix(y_true, y_pred_knn, labels=['organik', 'anorganik'])
        cm_svm = confusion_matrix(y_true, y_pred_svm, labels=['organik', 'anorganik'])
        
        # Tetapkan label kelas secara eksplisit
        class_labels = ['organik', 'anorganik']
        
        # Plot KNN confusion matrix
        display_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_labels)
        display_knn.plot(ax=ax[0], values_format='.0f', colorbar=True)
        ax[0].set_title(f"{nama} - KNN Confusion Matrix")
        ax[0].set_xlabel('Predicted Label')
        ax[0].set_ylabel('True Label')
        
        # Plot SVM confusion matrix
        display_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_labels)
        display_svm.plot(ax=ax[1], values_format='.0f', colorbar=True)
        ax[1].set_title(f"{nama} - SVM Confusion Matrix")
        ax[1].set_xlabel('Predicted Label')
        ax[1].set_ylabel('True Label')
        
        # Add overall accuracy as a subtitle
        acc_knn = (cm_knn[0,0] + cm_knn[1,1]) / cm_knn.sum() * 100
        acc_svm = (cm_svm[0,0] + cm_svm[1,1]) / cm_svm.sum() * 100
        plt.figtext(0.25, -0.05, f'KNN Accuracy: {acc_knn:.1f}%', ha='center')
        plt.figtext(0.75, -0.05, f'SVM Accuracy: {acc_svm:.1f}%', ha='center')
        
        plt.suptitle(f'Confusion Matrices for {nama} Features', y=1.05, fontsize=14)
        plt.tight_layout()
        plt.show()

    # --- Visualisasi Sample Images ---
    print("\n[INFO] Menampilkan visualisasi ekstraksi fitur...")
    sample_images = []
    
    # Ambil 1 sampel organik
    folder_organik = os.path.join(dataset_path, "organik")
    for img in os.listdir(folder_organik):
        sample_images.append((os.path.join(folder_organik, img), "organik"))
        break

    # Ambil 1 sampel anorganik
    folder_anorganik = os.path.join(dataset_path, "anorganik")
    for img in os.listdir(folder_anorganik):
        sample_images.append((os.path.join(folder_anorganik, img), "anorganik"))
        break

    # Visualisasi untuk setiap sampel
    for img_path, label in sample_images:
        print(f"\nVisualisasi untuk {os.path.basename(img_path)} ({label})")
        visualisasi_ekstraksi(img_path)

    # Hasil klasifikasi untuk sampel
    print("\nHasil Klasifikasi untuk Sample Images:")
    print("| Gambar | Label | Bentuk-KNN | Bentuk-SVM | Tekstur-KNN | Tekstur-SVM | Warna-KNN | Warna-SVM |")
    print("|---------|--------|------------|------------|--------------|--------------|-----------|-----------|")
    
    for img_path, label in sample_images:
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

    # Simpan hasil visualisasi untuk sampel terpilih
    print("\n[INFO] Menyimpan hasil visualisasi...")
    output_folder = os.path.join(base_dir, "dataset", "hasil")
    os.makedirs(output_folder, exist_ok=True)    # Ambil 10 sampel untuk setiap kategori
    anorganik_files = [f for f in os.listdir(os.path.join(dataset_path, "anorganik"))][:10]
    organik_files = [f for f in os.listdir(os.path.join(dataset_path, "organik"))][:10]

    # Proses dan simpan visualisasi
    for i, file in enumerate(anorganik_files):
        img_path = os.path.join(dataset_path, "anorganik", file)
        simpan_visualisasi(img_path, output_folder, "anorganik", i+1)

    for i, file in enumerate(organik_files):
        img_path = os.path.join(dataset_path, "organik", file)
        simpan_visualisasi(img_path, output_folder, "organik", i+1)

    print("[INFO] Visualisasi berhasil disimpan di folder:", output_folder)

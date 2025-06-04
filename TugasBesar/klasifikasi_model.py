import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def klasifikasi_knn(X, y, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("[KNN] Akurasi:", acc)
    print(classification_report(y_test, y_pred))
    return model

def klasifikasi_svm(X, y, kernel='rbf'):
    model = SVC(kernel=kernel)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("[SVM] Akurasi:", acc)
    return model

def prediksi_single_image(model, fitur):
    fitur = np.array(fitur).reshape(1, -1)
    return model.predict(fitur)[0]

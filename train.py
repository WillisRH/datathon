# import os
# import joblib
# import pandas as pd
# import numpy as np
# from scipy import sparse  # <-- ADDED: For memory-efficient sparse matrices
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, mean_squared_error
# from sklearn.model_selection import train_test_split

# DATA_DIR = "datasets"
# MODEL_PATH = "models/universal_business_model.pkl"

# os.makedirs("models", exist_ok=True)

# def process_dataset(file_path):
#     """Reads, cleans, and processes a single CSV file into features (X) and a target (y)."""
#     try:
#         df = pd.read_csv(file_path, on_bad_lines='skip') # Added on_bad_lines for robustness
#         if df.shape[0] < 10 or df.shape[1] < 2:
#             return None, None, None

#         df = df.dropna()
#         if df.shape[0] < 10:
#             return None, None, None

#         for col in df.select_dtypes(include='object').columns:
#             try:
#                 df[col] = LabelEncoder().fit_transform(df[col].astype(str))
#             except:
#                 continue

#         df = df.select_dtypes(include='number')
#         if df.shape[1] < 2:
#             return None, None, None

#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values

#         if len(set(y)) < 2:
#             return None, None, None

#         task = "classification" if len(set(y)) < 20 and pd.api.types.is_integer_dtype(y) else "regression"
#         return X, y, task
#     except Exception as e:
#         print(f"❌ Failed processing {file_path}: {e}")
#         return None, None, None

# def train_model():
#     """
#     Trains a model by collecting data from all CSVs, standardizing the feature space
#     using memory-efficient sparse matrices, and then fitting a single model.
#     """
#     print("\n🧠 Training model from datasets...")
#     X_pool, y_pool, task_pool = [], [], []

#     for filename in os.listdir(DATA_DIR):
#         if not filename.endswith(".csv"):
#             continue
#         path = os.path.join(DATA_DIR, filename)
#         X, y, task = process_dataset(path)
#         if X is None or X.shape[0] <= 5 or X.ndim != 2 or X.shape[1] == 0:
#             continue

#         X_pool.append(X)
#         y_pool.append(y)
#         task_pool.append(task)
#         print(f"✅ Collected from {filename} for {task} with {X.shape[1]} features.")

#     if not X_pool:
#         print("⚠️ No valid data found.")
#         return None

#     max_features = max(x.shape[1] for x in X_pool)
#     print(f"✨ Standardizing all datasets to {max_features} features using sparse matrices.")

#     # --- MEMORY ERROR FIX: Use sparse matrices for padding ---
#     X_sparse_pool = []
#     y_filtered_pool = []

#     for i, x in enumerate(X_pool):
#         # Convert numpy array to a sparse matrix
#         sparse_x = sparse.csr_matrix(x)
        
#         # Pad the sparse matrix to the right with zeros if needed
#         padding_width = max_features - sparse_x.shape[1]
#         if padding_width > 0:
#             # Create a sparse padding matrix (takes up almost no memory)
#             sparse_padding = sparse.csr_matrix((sparse_x.shape[0], padding_width))
#             # Horizontally stack the data and the padding
#             x_padded = sparse.hstack([sparse_x, sparse_padding], format='csr')
#             X_sparse_pool.append(x_padded)
#         else:
#             X_sparse_pool.append(sparse_x)
        
#         y_filtered_pool.append(y_pool[i])
#     # --- END OF FIX ---
    
#     if not X_sparse_pool:
#         print("⚠️ No data left after filtering.")
#         return None

#     # Vertically stack the list of sparse matrices into one large matrix
#     X_total = sparse.vstack(X_sparse_pool, format='csr')
#     y_total = np.concatenate(y_filtered_pool)
    
#     print(f"✅ Created final sparse training matrix with shape {X_total.shape} and {X_total.nnz} non-zero elements.")

#     task_types = set(task_pool)
#     if "regression" in task_types or len(set(y_total)) >= 20:
#         print("➡️  Training a REGRESSION model.")
#         model = HistGradientBoostingRegressor()
#     else:
#         print("➡️  Training a CLASSIFICATION model.")
#         model = HistGradientBoostingClassifier()

#     model.fit(X_total, y_total)
#     joblib.dump(model, MODEL_PATH)
#     print(f"✅ Model trained and saved to {MODEL_PATH}")
#     return model

# def load_or_train_model():
#     """Loads a pre-existing model or trains a new one if not found."""
#     if os.path.exists(MODEL_PATH):
#         print(f"✅ Loaded model from {MODEL_PATH}")
#         return joblib.load(MODEL_PATH)
#     else:
#         return train_model()

# def evaluate_model(model):
#     """Evaluates the trained model on each dataset individually."""
#     print("\n📊 Evaluating model on all datasets...\n")
    
#     n_features_expected = model.n_features_in_
#     is_classifier = isinstance(model, HistGradientBoostingClassifier)
#     print(f"ℹ️ Model expects {n_features_expected} features as input.")
    
#     for filename in os.listdir(DATA_DIR):
#         if not filename.endswith(".csv"):
#             continue
#         path = os.path.join(DATA_DIR, filename)
#         X, y, task_heuristic = process_dataset(path)
#         if X is None:
#             continue

#         # Pad or truncate the evaluation data to match the model's expectation
#         if X.shape[1] < n_features_expected:
#             padding = np.zeros((X.shape[0], n_features_expected - X.shape[1]))
#             X = np.hstack((X, padding))
#         elif X.shape[1] > n_features_expected:
#             X = X[:, :n_features_expected]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         try:
#             y_pred = model.predict(X_test)
#             print(f"📁 Dataset: {filename} | Heuristic Task: {task_heuristic}")
            
#             # Use metric based on the type of model that was actually trained
#             if is_classifier:
#                 print(classification_report(y_test, y_pred, zero_division=0))
#             else:
#                 mse = mean_squared_error(y_test, y_pred)
#                 print(f"🧮 MSE: {mse:.4f}")

#         except Exception as e:
#             print(f"❌ Error evaluating {filename}: {e}")
#         print("-" * 50)

# def try_predict(model):
#     """Performs a single prediction on dummy data to ensure the model is working."""
#     print("\n🤖 Trying a prediction with dummy input...\n")
#     try:
#         n_features = model.n_features_in_
#         # Create a dense numpy array for the single prediction
#         dummy = np.random.rand(1, n_features)
#         pred = model.predict(dummy)
#         print(f"✅ Prediction from {n_features}-feature dummy input:\n→ {pred.tolist()}")
#     except Exception as e:
#         print(f"❌ Failed to predict: {e}")

# if __name__ == "__main__":
#     model = load_or_train_model()
#     if model:
#         evaluate_model(model)
#         try_predict(model)


import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
DATA_DIR = "datasets_behavioral"
MODEL_PATH = "models/behavioral_incentives_model.pkl"

# ==============================================================================
# BAGIAN 1: PEMBUATAN DATASET SINTETIS (Simulasi Ekonomi Perilaku)
# ==============================================================================

def generate_datasets():
    """
    Membuat beberapa dataset sintetis yang mensimulasikan pengaruh
    insentif finansial dan non-finansial terhadap kinerja karyawan.
    """
    print("🌱 Generating synthetic datasets based on Behavioral Economics principles...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Dataset 1: Startup Culture (Fokus Non-Finansial) ---
    n_samples = 200
    data_startup = {
        'gaji_bulanan': np.random.normal(10, 2, n_samples).round(2), # dalam juta Rupiah
        'jam_kerja_fleksibel': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]), # 1=Ya, 0=Tidak (Otonomi)
        'dapat_pujian_publik': np.random.choice([1, 0], n_samples, p=[0.7, 0.3]), # 1=Ya, 0=Tidak (Pengakuan)
        'peluang_training': np.random.randint(1, 5, n_samples), # (Pengembangan Diri)
        'lingkungan_kerja': ['Kolaboratif', 'Mendukung', 'Biasa'] * (n_samples // 3 + 1)
    }
    df_startup = pd.DataFrame(data_startup)
    # Formula Kinerja: Sangat dipengaruhi oleh insentif non-finansial
    kinerja_startup = (
        2 +
        df_startup['gaji_bulanan'] * 0.1 +
        df_startup['jam_kerja_fleksibel'] * 1.5 +
        df_startup['dapat_pujian_publik'] * 1.8 +
        df_startup['peluang_training'] * 0.4 +
        np.random.normal(0, 0.5, n_samples) # noise
    ).clip(1, 10).round(2)
    df_startup['kinerja_karyawan'] = kinerja_startup
    df_startup.to_csv(os.path.join(DATA_DIR, "startup_culture.csv"), index=False)
    print(f"✅ Created {os.path.join(DATA_DIR, 'startup_culture.csv')}")


    # --- Dataset 2: Corporate Sales (Fokus Finansial) ---
    n_samples = 250
    data_corporate = {
        'gaji_bulanan': np.random.normal(20, 5, n_samples).round(2),
        'bonus_tahunan': np.random.normal(50, 15, n_samples).round(2), # dalam juta Rupiah
        'target_tercapai': np.random.choice([1, 0], n_samples, p=[0.6, 0.4]),
        'jam_kerja_fleksibel': np.random.choice([1, 0], n_samples, p=[0.1, 0.9]), # Jarang ada
    }
    df_corporate = pd.DataFrame(data_corporate)
    # Formula Kinerja: Sangat dipengaruhi oleh insentif finansial
    kinerja_corporate = (
        1 +
        df_corporate['gaji_bulanan'] * 0.2 +
        df_corporate['bonus_tahunan'] * 0.05 +
        df_corporate['target_tercapai'] * 2.5 +
        df_corporate['jam_kerja_fleksibel'] * 0.5 + # Pengaruh kecil
        np.random.normal(0, 0.6, n_samples) # noise
    ).clip(1, 10).round(2)
    df_corporate['kinerja_karyawan'] = kinerja_corporate
    df_corporate.to_csv(os.path.join(DATA_DIR, "corporate_sales.csv"), index=False)
    print(f"✅ Created {os.path.join(DATA_DIR, 'corporate_sales.csv')}")


    # --- Dataset 3: Support Team (Fokus Lingkungan Kerja) ---
    n_samples = 150
    data_support = {
        'gaji_bulanan': np.random.normal(8, 1.5, n_samples).round(2),
        'skor_kolaborasi_tim': np.random.randint(1, 6, n_samples), # Skala 1-5 (Lingkungan Positif)
        'skor_feedback_manajer': np.random.randint(1, 6, n_samples), # Skala 1-5 (Apresiasi)
        'beban_kerja': ['Rendah', 'Normal', 'Tinggi'] * (n_samples // 3 + 1)
    }
    df_support = pd.DataFrame(data_support)
    # Formula Kinerja: Sangat dipengaruhi oleh faktor sosial dan lingkungan
    kinerja_support = (
        3 +
        df_support['gaji_bulanan'] * 0.15 +
        df_support['skor_kolaborasi_tim'] * 0.8 +
        df_support['skor_feedback_manajer'] * 0.9 +
        np.random.normal(0, 0.4, n_samples) # noise
    ).clip(1, 10).round(2)
    df_support['kinerja_karyawan'] = kinerja_support
    df_support.to_csv(os.path.join(DATA_DIR, "support_team.csv"), index=False)
    print(f"✅ Created {os.path.join(DATA_DIR, 'support_team.csv')}")


# ==============================================================================
# BAGIAN 2: KODE UNIVERSAL MODEL (Kode Asli Anda, Disesuaikan)
# ==============================================================================

def process_dataset(file_path):
    """Membaca, membersihkan, dan memproses satu file CSV menjadi fitur (X) dan target (y)."""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        if df.shape[0] < 10 or df.shape[1] < 2:
            return None, None, None

        # Menggunakan encoder yang disimpan untuk konsistensi, tapi untuk demo ini kita buat baru
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            except:
                continue

        df = df.select_dtypes(include='number')
        if df.shape[1] < 2:
            return None, None, None

        # Asumsi kolom terakhir adalah target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if len(set(y)) < 2:
            return None, None, None

        # Heuristik untuk menentukan jenis tugas (klasifikasi/regresi)
        task = "classification" if len(set(y)) < 20 and pd.api.types.is_integer_dtype(y) else "regression"
        return X, y, task
    except Exception as e:
        print(f"❌ Failed processing {file_path}: {e}")
        return None, None, None

def train_model():
    """
    Melatih model dengan mengumpulkan data dari semua CSV, menstandarkan ruang fitur
    menggunakan sparse matrices yang efisien memori, lalu melatih satu model.
    """
    print("\n🧠 Training model from datasets...")
    X_pool, y_pool, task_pool = [], [], []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(DATA_DIR, filename)
        X, y, task = process_dataset(path)
        if X is None or X.shape[0] <= 5 or X.ndim != 2 or X.shape[1] == 0:
            continue

        X_pool.append(X)
        y_pool.append(y)
        task_pool.append(task)
        print(f"✅ Collected from {filename} for {task} with {X.shape[1]} features.")

    if not X_pool:
        print("⚠️ No valid data found.")
        return None

    max_features = max(x.shape[1] for x in X_pool)
    print(f"✨ Standardizing all datasets to {max_features} features using sparse matrices.")

    X_sparse_pool = []
    y_filtered_pool = []

    for i, x in enumerate(X_pool):
        sparse_x = sparse.csr_matrix(x)
        padding_width = max_features - sparse_x.shape[1]
        if padding_width > 0:
            sparse_padding = sparse.csr_matrix((sparse_x.shape[0], padding_width))
            x_padded = sparse.hstack([sparse_x, sparse_padding], format='csr')
            X_sparse_pool.append(x_padded)
        else:
            X_sparse_pool.append(sparse_x)
        y_filtered_pool.append(y_pool[i])

    if not X_sparse_pool:
        print("⚠️ No data left after filtering.")
        return None

    X_total = sparse.vstack(X_sparse_pool, format='csr')
    y_total = np.concatenate(y_filtered_pool)
    
    print(f"✅ Created final sparse training matrix with shape {X_total.shape} and {X_total.nnz} non-zero elements.")

    # Karena target kita adalah skor kinerja (kontinu), kita akan paksakan Regresi
    print("➡️  Training a REGRESSION model to predict employee performance.")
    model = HistGradientBoostingRegressor(random_state=42)
    
    model.fit(X_total, y_total)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
    return model

def load_or_train_model():
    """Memuat model yang sudah ada atau melatih yang baru jika tidak ditemukan."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loaded model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        return train_model()

def evaluate_model(model):
    """Mengevaluasi model yang telah dilatih pada setiap dataset secara individual."""
    print("\n📊 Evaluating model on all datasets...\n")
    
    n_features_expected = model.n_features_in_
    is_classifier = isinstance(model, HistGradientBoostingClassifier)
    print(f"ℹ️ Model expects {n_features_expected} features as input.")
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(DATA_DIR, filename)
        X, y, task_heuristic = process_dataset(path)
        if X is None:
            continue

        # Pad atau potong data evaluasi agar sesuai dengan ekspektasi model
        if X.shape[1] < n_features_expected:
            padding = np.zeros((X.shape[0], n_features_expected - X.shape[1]))
            X = np.hstack((X, padding))
        elif X.shape[1] > n_features_expected:
            X = X[:, :n_features_expected]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            y_pred = model.predict(X_test)
            print(f"📁 Dataset: {filename} | Heuristic Task: {task_heuristic}")
            
            if is_classifier:
                print(classification_report(y_test, y_pred, zero_division=0))
            else: # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"   🧮 Mean Squared Error: {mse:.4f}")
                print(f"   🎯 R-squared Score: {r2:.4f} (semakin dekat ke 1, semakin baik)")

        except Exception as e:
            print(f"❌ Error evaluating {filename}: {e}")
        print("-" * 60)

def try_predict(model):
    """Melakukan prediksi tunggal pada data dummy untuk memastikan model berfungsi."""
    print("\n🤖 Trying a prediction with dummy input...\n")
    try:
        n_features = model.n_features_in_
        # Membuat data dummy yang realistis untuk skenario startup
        # Fitur: gaji, jam_fleksibel, pujian_publik, peluang_training
        dummy_startup = np.array([[12, 1, 1, 3]]) 
        
        # Padding agar sesuai dengan input model
        if dummy_startup.shape[1] < n_features:
            padding = np.zeros((dummy_startup.shape[0], n_features - dummy_startup.shape[1]))
            dummy_startup = np.hstack((dummy_startup, padding))

        pred = model.predict(dummy_startup)
        print(f"✅ Prediction for a high-performing 'startup' employee:")
        print(f"   Input: Gaji=12jt, Fleksibel=Ya, Pujian=Ya, Training=3")
        print(f"   → Predicted Performance Score: {pred[0]:.2f} (out of 10)")
    except Exception as e:
        print(f"❌ Failed to predict: {e}")

# ==============================================================================
# BAGIAN 3: EKSEKUSI UTAMA
# ==============================================================================

if __name__ == "__main__":
    # Langkah 1: Buat dataset jika belum ada
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        generate_datasets()
    else:
        print("✅ Datasets already exist. Skipping generation.")

    # Langkah 2: Muat atau latih model
    model = load_or_train_model()
    
    # Langkah 3: Jika model berhasil dibuat/dimuat, evaluasi dan coba prediksi
    if model:
        evaluate_model(model)
        try_predict(model)
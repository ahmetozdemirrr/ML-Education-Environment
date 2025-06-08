# backend/app/data_processor.py - Enhanced version
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Any, Optional


def _find_dataset_filepath_in_processor(dataset_id: str, data_dir: str) -> Optional[str]:
    """Verilen dataset ID'sine karşılık gelen dosya yolunu bulur (CSV veya JSON)."""
    if ".." in dataset_id or "/" in dataset_id or "\\" in dataset_id:
        print(f"Uyarı: Güvenlik riski oluşturan dataset ID formatı: {dataset_id}")
        return None

    possible_extensions = [".csv", ".json"]
    for ext in possible_extensions:
        filepath = os.path.join(data_dir, f"{dataset_id}{ext}")
        if os.path.exists(filepath):
            return filepath
    print(f"Dosya bulunamadı: {dataset_id} (Aranan yer: {data_dir})")
    return None


"""
    Dataseti yükler, X ve y olarak ayırır.
    Global ayarlara göre train/test split uygular.
    Özellik ölçeklendirmeyi (StandardScaler/MinMaxScaler) train/test split sonrası doğru şekilde uygular.
    Cross-validation seçiliyse, tüm veriyi (ölçeklenmemiş X ve ölçeklenmemiş y) döndürür,
    çünkü ölçeklendirme Pipeline içinde her katlamada yapılmalıdır.
"""
def load_and_prepare_data(
    dataset_id: str,
    data_dir: str,
    global_settings: Dict[str, Any]
) -> Dict[str, Any]:

    print(f"Dataset işleniyor: {dataset_id}. Global Ayarlar: {global_settings}")

    filepath = _find_dataset_filepath_in_processor(dataset_id, data_dir)

    if not filepath:
        raise FileNotFoundError(f"Dataset dosyası bulunamadı: '{dataset_id}' (Aranan yer: {data_dir})")

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath, orient='records')
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {filepath}")
    except Exception as e:
        raise ValueError(f"Dataset ({dataset_id}) okunurken hata: {e}")

    if 'target' not in df.columns:
        raise ValueError(f"'target' sütunu '{dataset_id}' datasetinde bulunamadı. Lütfen dataseti kontrol edin.")

    X = df.drop('target', axis=1)
    y = df['target']

    # Sayısal olmayan sütunları kontrol et ve kaldır
    numeric_cols = X.select_dtypes(include=np.number).columns
    if len(numeric_cols) != X.shape[1]:
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        print(f"Uyarı: '{dataset_id}' datasetindeki sayısal olmayan özellikler ({list(non_numeric_cols)}) atılıyor.")
        X = X[numeric_cols]

    if X.empty:
        raise ValueError(f"Özellik matrisi (X) '{dataset_id}' için boş kaldı (sayısal olmayanlar atıldıktan sonra).")

    # Temel veri yapısını hazırla
    processed_data = {
        "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        "X_full": X.copy(), "y_full": y.copy(), # CV ve ölçeklenmemiş tam veri için
        "feature_names": list(X.columns),
        "data_preparation_log": [],
        "scaled": False # Ölçeklendirme yapılıp yapılmadığını takip etmek için
    }

    # Feature scaling ayarları (frontend'den gelmeli, varsayılan değerler)
    apply_scaling_setting = global_settings.get("applyFeatureScaling", True)
    scaler_type_setting = global_settings.get("scalerType", 'standard') # 'standard' veya 'minmax'

    scaler = None
    if apply_scaling_setting:
        if scaler_type_setting == 'minmax':
            scaler = MinMaxScaler()
        else: # Varsayılan veya 'standard'
            scaler = StandardScaler()

    # Train/Test Split durumu
    if global_settings.get('useTrainTestSplit', False):
        test_size = global_settings.get('testSplitRatio', 0.2)
        random_state_val = global_settings.get('randomSeed')
        random_state = int(random_state_val) if random_state_val is not None else None

        # Stratification için y'nin uygun olup olmadığını kontrol et
        stratify_option = y if y.nunique() > 1 and y.nunique() < len(y) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_option
        )

        # Ölçeklendirme uygula (eğer seçiliyse)
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            processed_data["X_train"] = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            processed_data["X_test"] = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            processed_data["scaled"] = True
            log_msg = f"Train/Test split (test_size={test_size}, random_state={random_state}). Özellikler {scaler_type_setting}Scaler ile ölçeklendi."
        else:
            processed_data["X_train"] = X_train.copy()
            processed_data["X_test"] = X_test.copy()
            log_msg = f"Train/Test split (test_size={test_size}, random_state={random_state}). Ölçeklendirme uygulanmadı."

        processed_data["y_train"] = y_train.copy()
        processed_data["y_test"] = y_test.copy()
        processed_data["data_preparation_log"].append(log_msg)

        # CV için kullanılan X_full ve y_full'u bu durumda None yapalım ki karışmasın
        processed_data["X_full"] = None
        processed_data["y_full"] = None

    # Cross-Validation durumu
    elif global_settings.get('useCrossValidation', False):
        # Cross-validation için, ölçeklendirme Pipeline içinde yapılmalıdır.
        # Bu fonksiyon ölçeklenmemiş X_full ve y_full döndürür.
        # Model eğitim aşamasında Pipeline kullanılacaksa, ölçekleyici bu Pipeline'a eklenir.
        if scaler:
            # Eğer burada ölçeklendirme isteniyorsa (Pipeline kullanılmayacaksa), tüm veriyi ölçekle
            # Ama bu CV'de veri sızıntısına neden olabilir, o yüzden uyarı verelim
            X_scaled_full = scaler.fit_transform(X)
            processed_data["X_full"] = pd.DataFrame(X_scaled_full, columns=X.columns, index=X.index)
            processed_data["scaled"] = True
            processed_data["data_preparation_log"].append(
                f"Cross-validation için tüm veri. Özellikler {scaler_type_setting}Scaler ile ölçeklendi "
                f"(UYARI: Bu, CV'de veri sızıntısına neden olabilir. Pipeline kullanılması önerilir)."
            )
        else:
            # Ölçeklendirme kapalıysa, orijinal X_full kalır
            processed_data["data_preparation_log"].append(
                "Cross-validation için tüm veri. Ölçeklendirme uygulanmadı."
            )
    else:
        # Hiçbir yöntem seçilmediyse (frontend bunu engellemeli)
        processed_data["data_preparation_log"].append(
            "Uyarı: Ne Train/Test Split ne de Cross-Validation seçilmedi. Veri ölçeklenmeden olduğu gibi bırakıldı."
        )
        # X_full ve y_full zaten orijinal veriyi içeriyor.

    print(f"'{dataset_id}' başarıyla yüklendi ve ön işlendi. {processed_data['data_preparation_log']}")
    return processed_data

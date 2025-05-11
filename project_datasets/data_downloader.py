import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def save_dataframe_to_csv(df, filename, target_dir="./project_datasets/datasets"):
    """DataFrame'i belirtilen dizine CSV olarak kaydeder."""
    filepath = os.path.join(target_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset '{filename}' başarıyla '{filepath}' olarak kaydedildi.")

def main():
    print("=" * 40)

    # 1. Iris Dataset
    try:
        iris = load_iris()
        df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df_iris["target"] = iris.target
        save_dataframe_to_csv(df_iris, "iris.csv")
    
    except Exception as e:
        print(f"Hata (Iris Dataset): {e}")
    print("-" * 30)

    # 2. Two Moons Dataset
    try:
        X_moons, y_moons = make_moons(n_samples=400, noise=0.15, random_state=42)
        df_moons = pd.DataFrame(X_moons, columns=["feature_1", "feature_2"])
        df_moons["target"] = y_moons
        save_dataframe_to_csv(df_moons, "two_moons_data.csv")
    
    except Exception as e:
        print(f"Hata (Two Moons Dataset): {e}")
    print("-" * 30)

    # 3. Wine Dataset
    try:
        wine = load_wine()
        df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
        df_wine["target"] = wine.target
        save_dataframe_to_csv(df_wine, "wine_data.csv")
    
    except Exception as e:
        print(f"Hata (Wine Dataset): {e}")
    print("-" * 30)

    # 4. Breast Cancer Wisconsin (Diagnostic) Dataset
    try:
        cancer = load_breast_cancer()
        df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df_cancer["target"] = cancer.target
        df_cancer.columns = ["_".join(col.lower().split()).replace('(', '').replace(')', '') for col in df_cancer.columns]
        save_dataframe_to_csv(df_cancer, "breast_cancer_wisconsin.csv")
    
    except Exception as e:
        print(f"Hata (Breast Cancer Dataset): {e}")
    print("-" * 30)

    # 5. Digits Dataset
    try:
        digits = load_digits()
        feature_names_digits = [f"pixel_{i:02d}" for i in range(digits.data.shape[1])]
        df_digits = pd.DataFrame(data=digits.data, columns=feature_names_digits)
        df_digits["target"] = digits.target
        save_dataframe_to_csv(df_digits, "digits_data.csv")
    
    except Exception as e:
        print(f"Hata (Digits Dataset): {e}")
    print("-" * 30)

    # 6. Haberman's Survival Dataset
    try:
        url_haberman = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
        df_haberman = pd.read_csv(url_haberman, header=None, names=['age', 'op_year', 'axil_nodes', 'target'])
        df_haberman['target'] = df_haberman['target'].replace({1: 0, 2: 1})
        save_dataframe_to_csv(df_haberman, "haberman_survival.csv")
    
    except Exception as e:
        print(f"Hata (Haberman's Survival Dataset): {e}")
    print("-" * 30)

    # 7. Pima Indians Diabetes Dataset
    try:
        url_pima = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df_pima = pd.read_csv(url_pima)
        df_pima.columns = [col.lower().replace(' ', '_') for col in df_pima.columns]
        if 'outcome' in df_pima.columns:
            df_pima.rename(columns={'outcome': 'target'}, inplace=True)
        save_dataframe_to_csv(df_pima, "pima_indians_diabetes.csv")
    
    except Exception as e:
        print(f"Hata (Pima Indians Diabetes Dataset): {e}")
    print("-" * 30)

    # 8. Banknote Authentication Dataset
    try:
        url_banknote = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        df_banknote = pd.read_csv(url_banknote, header=None, names=['variance', 'skewness', 'curtosis', 'entropy', 'target'])
        save_dataframe_to_csv(df_banknote, "banknote_authentication.csv")
    
    except Exception as e:
        print(f"Hata (Banknote Authentication Dataset): {e}")
    print("-" * 30)

    # 9. Mushroom Dataset (Tümü Kategorik)
    try:
        url_mushroom = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        column_names_mushroom = ['target'] + [f'feature_{i}' for i in range(1, 23)]
        df_mushroom = pd.read_csv(url_mushroom, header=None, names=column_names_mushroom, na_values=['?'])
        for col in df_mushroom.columns:
            if df_mushroom[col].isnull().any():
                df_mushroom[col] = df_mushroom[col].fillna(df_mushroom[col].mode()[0])
        encoder_mushroom = OrdinalEncoder()
        df_mushroom_encoded = pd.DataFrame(encoder_mushroom.fit_transform(df_mushroom), columns=df_mushroom.columns)
        save_dataframe_to_csv(df_mushroom_encoded, "mushroom_data.csv")
    
    except Exception as e:
        print(f"Hata (Mushroom Dataset): {e}")
    print("-" * 30)

    # 10. Synthetic Classification Dataset (Daha Zorlu)
    try:
        X_synth, y_synth = make_classification(
            n_samples=1000, n_features=20, n_informative=12, n_redundant=3, n_repeated=0,
            n_classes=3, n_clusters_per_class=2, weights=None, flip_y=0.05, random_state=42 )
        feature_names_synth = [f'feature_{i}' for i in range(X_synth.shape[1])]
        df_synth = pd.DataFrame(X_synth, columns=feature_names_synth)
        df_synth['target'] = y_synth
        save_dataframe_to_csv(df_synth, "synthetic_classification_hard.csv")
    
    except Exception as e:
        print(f"Hata (Synthetic Classification Dataset): {e}")
    print("-" * 30)

    # 11. Abalone Dataset (Sınıflandırma için dönüştürüldü)
    try:
        url_abalone = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        abalone_cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
        df_abalone = pd.read_csv(url_abalone, header=None, names=abalone_cols)
        encoder_sex = OrdinalEncoder() # 'M', 'F', 'I' için
        df_abalone['sex'] = encoder_sex.fit_transform(df_abalone[['sex']])
        bins = [0, 8, 10, np.inf]
        labels = [0, 1, 2] # genç, yetişkin, yaşlı
        df_abalone['target'] = pd.cut(df_abalone['rings'], bins=bins, labels=labels, right=True)
        df_abalone.drop('rings', axis=1, inplace=True)
        df_abalone.dropna(subset=['target'], inplace=True) # pd.cut sonrası NaN oluşursa
        df_abalone['target'] = df_abalone['target'].astype(int)
        save_dataframe_to_csv(df_abalone, "abalone_classification.csv")
    
    except Exception as e:
        print(f"Hata (Abalone Dataset): {e}")
    print("-" * 30)

    # 12. Bike Sharing Hourly Dataset (Sınıflandırma için dönüştürülecek)
    try:
        url_bike = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
        print("Bike Sharing dataseti indiriliyor (zip dosyası)...")

        response = requests.get(url_bike)
        response.raise_for_status() # HTTP hatalarını kontrol eder

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            if 'hour.csv' in z.namelist():
                with z.open('hour.csv') as f:
                    df_bike_hourly = pd.read_csv(f, sep=',', header=0)
            else:
                raise FileNotFoundError("'hour.csv' dosyası zip arşivinde bulunamadı.")

        cols_to_use_and_target = [
            'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt'
        ]
        df_bike_hourly_filtered = df_bike_hourly[[col for col in cols_to_use_and_target if col in df_bike_hourly.columns]].copy()

        if 'cnt' not in df_bike_hourly_filtered.columns:
            raise ValueError("'cnt' sütunu Bike Sharing datasetinde işlendikten sonra bulunamadı.")

        df_bike_hourly_filtered['target'] = pd.qcut(df_bike_hourly_filtered['cnt'], q=3, labels=[0, 1, 2], duplicates='drop')
        df_bike_hourly_filtered.drop('cnt', axis=1, inplace=True) 

        new_cols = [col.lower().replace(' ', '_') for col in df_bike_hourly_filtered.columns if col != 'target']
        df_bike_hourly_filtered.columns = new_cols + ['target']

        save_dataframe_to_csv(df_bike_hourly_filtered, "bike_sharing_hourly_classification.csv")
    
    except Exception as e:
        print(f"Hata (Bike Sharing Dataset): {e}")
        print("Not: Bike Sharing dataseti için internet bağlantısı ve zip dosyasından okuma gereklidir.")
    print("-" * 30)

    # 13. MNIST Dataset (Tamamı ve Alt Kümesi) ---
    print("MNIST dataseti indiriliyor (bu işlem UZUN sürebilir)...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto', cache=True)
        df_mnist_full = mnist.frame.copy() # Orijinal DataFrame'i kopyala
        
        if 'class' in df_mnist_full.columns:
            df_mnist_full.rename(columns={'class': 'target'}, inplace=True)
        elif 'target' not in df_mnist_full.columns:
            raise ValueError("MNIST tam verisetinde 'class' veya 'target' sütunu bulunamadı.")
        
        df_mnist_full['target'] = df_mnist_full['target'].astype(int)
        
        # 1. MNIST Tamamı (70k)
        save_dataframe_to_csv(df_mnist_full, "mnist_full_70k.csv")
        print("-" * 15)

        # 2. MNIST Alt Kümesi (örn: 7k)
        n_samples_subset = 7000
        print(f"MNIST datasetinin {n_samples_subset} örneklik alt kümesi oluşturuluyor...")
        X_mnist_all = df_mnist_full.drop('target', axis=1)
        y_mnist_all = df_mnist_full['target']

        _, X_mnist_subset, _, y_mnist_subset = train_test_split(
            X_mnist_all, y_mnist_all, 
            test_size=n_samples_subset,
            stratify=y_mnist_all, 
            random_state=42
        )        
        df_mnist_subset = pd.DataFrame(X_mnist_subset)
        df_mnist_subset['target'] = y_mnist_subset.values
        
        save_dataframe_to_csv(df_mnist_subset, f"mnist_subset_7k.csv")

    except Exception as e:
        print(f"Hata (MNIST Dataset): {e}")
        print("Not: MNIST dataseti için internet bağlantısı gereklidir ve indirme/işleme çok uzun sürebilir.")
    print("=" * 40)

    print("Tüm dataset işlemleri tamamlandı.")
    print(f"Lütfen '{os.getcwd()}' dizinini kontrol edin.")

if __name__ == "__main__":
    main()

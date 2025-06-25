import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DatasetVisualizer:
    def __init__(self, datasets_path: str = "/app/mounted_datasets"):
        self.datasets_path = datasets_path

        # MAPPING: Frontend tarafından istenen isim -> Gerçek dosya ismi
        self.filename_mapping = {
            # MNIST datasets
            "mnist_digits_(full_70k).csv": "mnist_full_70k.csv",
            "mnist_digits_%28full_70k%29.csv": "mnist_full_70k.csv",  # URL encoded version
            "mnist_digits_(subset_7k).csv": "mnist_subset_7k.csv",
            "mnist_digits_%28subset_7k%29.csv": "mnist_subset_7k.csv",  # URL encoded version

            # Bike sharing
            "bike_sharing_hourly_demand.csv": "bike_sharing_hourly_classification.csv",

            # Abalone
            "abalone_age_(classification).csv": "abalone_classification.csv",
            "abalone_age_%28classification%29.csv": "abalone_classification.csv",  # URL encoded version

            # Synthetic
            "synthetic_classification.csv": "synthetic_classification_hard.csv",

            # Mushroom
            "mushroom_dataset.csv": "mushroom_data.csv",

            # Haberman
            "haberman's_survival_data.csv": "haberman_survival.csv",
            "haberman%27s_survival_data.csv": "haberman_survival.csv",  # URL encoded version

            # Two moons
            "two_moons_(synthetic).csv": "two_moons_data.csv",
            "two_moons_%28synthetic%29.csv": "two_moons_data.csv",  # URL encoded version
        }

        self.dataset_configs = {
            "iris.csv": {
                "target_column": "class",
                "name": "Iris Dataset",
                "description": "Classic flower classification dataset with 4 features",
                "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            },
            "wine_data.csv": {
                "target_column": "class",
                "name": "Wine Dataset",
                "description": "Wine classification based on chemical analysis",
                "feature_columns": None  # Will auto-detect
            },
            "breast_cancer_wisconsin.csv": {
                "target_column": "diagnosis",
                "name": "Breast Cancer Wisconsin",
                "description": "Breast cancer diagnosis based on cell features",
                "feature_columns": None
            },
            "two_moons_data.csv": {
                "target_column": "target",
                "name": "Two Moons Dataset",
                "description": "Synthetic 2D classification dataset",
                "feature_columns": ["feature_0", "feature_1"]
            },
            "digits_data.csv": {
                "target_column": "target",
                "name": "Digits Dataset",
                "description": "Handwritten digit recognition dataset",
                "feature_columns": None
            },
            "pima_indians_diabetes.csv": {
                "target_column": "Outcome",
                "name": "Pima Indians Diabetes",
                "description": "Diabetes prediction based on medical factors",
                "feature_columns": None
            },
            "synthetic_classification_hard.csv": {
                "target_column": "target",
                "name": "Synthetic Hard Classification",
                "description": "Challenging synthetic classification dataset",
                "feature_columns": None
            },
            "banknote_authentication.csv": {
                "target_column": "class",
                "name": "Banknote Authentication",
                "description": "Banknote authentication based on image features",
                "feature_columns": None
            },
            "haberman_survival.csv": {
                "target_column": "survival_status",
                "name": "Haberman Survival",
                "description": "Survival of patients who had undergone surgery for breast cancer",
                "feature_columns": None
            },
            "mushroom_data.csv": {
                "target_column": "class",
                "name": "Mushroom Classification",
                "description": "Mushroom classification: edible or poisonous",
                "feature_columns": None
            },
            "abalone_classification.csv": {
                "target_column": "class",
                "name": "Abalone Classification",
                "description": "Abalone age classification based on physical measurements",
                "feature_columns": None
            },
            "bike_sharing_hourly_classification.csv": {
                "target_column": "cnt_class",
                "name": "Bike Sharing Classification",
                "description": "Bike sharing demand classification",
                "feature_columns": None
            },
            "mnist_subset_7k.csv": {
                "target_column": "label",
                "name": "MNIST Subset 7K",
                "description": "7000 sample subset of MNIST handwritten digits",
                "feature_columns": None
            },
            "mnist_full_70k.csv": {
                "target_column": "label",
                "name": "MNIST Full 70K",
                "description": "Complete MNIST dataset with 70000 handwritten digits",
                "feature_columns": None
            }
        }

    def resolve_filename(self, requested_filename: str) -> str:
        """
        Frontend'den gelen dosya ismini gerçek dosya ismine çevirir
        """
        # URL decode işlemi yapalım
        import urllib.parse
        decoded_filename = urllib.parse.unquote(requested_filename)

        # Mapping'de var mı kontrol et
        if decoded_filename in self.filename_mapping:
            actual_filename = self.filename_mapping[decoded_filename]
            logger.info(f"Mapped '{requested_filename}' -> '{actual_filename}'")
            return actual_filename

        # Orijinal isim mapping'de de var mı kontrol et
        if requested_filename in self.filename_mapping:
            actual_filename = self.filename_mapping[requested_filename]
            logger.info(f"Mapped '{requested_filename}' -> '{actual_filename}'")
            return actual_filename

        # Mapping'de yoksa, orijinal ismi döndür
        logger.info(f"No mapping found for '{requested_filename}', using as-is")
        return requested_filename

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset from CSV file"""
        try:
            # Dosya ismini resolve et
            actual_filename = self.resolve_filename(dataset_name)
            file_path = os.path.join(self.datasets_path, actual_filename)

            if not os.path.exists(file_path):
                logger.error(f"Dataset file not found: {file_path}")
                # Alternatif dosya isimlerini de deneyelim
                alternative_path = os.path.join(self.datasets_path, dataset_name)
                if os.path.exists(alternative_path):
                    logger.info(f"Found alternative file: {alternative_path}")
                    file_path = alternative_path
                else:
                    return None

            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset {actual_filename} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Preprocess dataset for visualization"""
        try:
            # Gerçek dosya ismini al
            actual_filename = self.resolve_filename(dataset_name)
            config = self.dataset_configs.get(actual_filename, {})
            target_column = config.get("target_column")

            if target_column and target_column in df.columns:
                # Separate features and target
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                # Try common target column names
                common_targets = ['target', 'class', 'label', 'y', 'outcome', 'diagnosis', 'cnt_class', 'survival_status']
                target_found = None

                for common_target in common_targets:
                    if common_target in df.columns:
                        target_found = common_target
                        break

                if target_found:
                    y = df[target_found]
                    X = df.drop(columns=[target_found])
                else:
                    # If no target column specified, use last column as target
                    y = df.iloc[:, -1]
                    X = df.iloc[:, :-1]

            # Remove non-numeric columns from features
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_columns]

            # Handle missing values
            X_numeric = X_numeric.fillna(X_numeric.mean())

            # Encode target if it's categorical
            le = LabelEncoder()
            if y.dtype == 'object' or y.dtype == 'bool':
                y_encoded = le.fit_transform(y)
                class_names = [str(name) for name in le.classes_.tolist()]  # Convert to Python strings
            else:
                y_encoded = y.values
                class_names = [str(name) for name in sorted(list(set(y_encoded)))]  # Convert to Python strings

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)

            return {
                "X": X_scaled,
                "y": y_encoded,
                "feature_names": X_numeric.columns.tolist(),
                "class_names": class_names,
                "original_shape": tuple(X_numeric.shape),  # Convert to tuple
                "n_samples": int(len(X_numeric)),  # Convert to int
                "n_features": int(len(X_numeric.columns)),  # Convert to int
                "n_classes": int(len(class_names))  # Convert to int
            }

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {e}")
            return None

    def reduce_dimensions(self, X: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality for visualization"""
        try:
            if X.shape[1] <= 2:
                # Already 2D or less
                if X.shape[1] == 1:
                    # Add a dummy second dimension
                    return np.column_stack([X, np.zeros(X.shape[0])])
                return X

            if method.lower() == "pca":
                reducer = PCA(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)
                logger.info(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
                return X_reduced

            elif method.lower() == "tsne":
                # t-SNE for non-linear dimensionality reduction
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X)//4))
                X_reduced = reducer.fit_transform(X)
                return X_reduced

            else:
                # Default to PCA
                reducer = PCA(n_components=n_components, random_state=42)
                return reducer.fit_transform(X)

        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            # Fallback: return first two features
            return X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(X.shape[0])])

    def generate_visualization_data(self, dataset_name: str, method: str = "pca") -> Dict:
        """Generate complete visualization data for a dataset"""
        try:
            # Load and preprocess dataset
            df = self.load_dataset(dataset_name)
            if df is None:
                return {"error": f"Could not load dataset: {dataset_name}"}

            processed = self.preprocess_dataset(df, dataset_name)
            if processed is None:
                return {"error": f"Could not preprocess dataset: {dataset_name}"}

            # Reduce dimensions
            X_2d = self.reduce_dimensions(processed["X"], method=method)

            # Create train/test split for visualization
            X_train, X_test, y_train, y_test = train_test_split(
                X_2d, processed["y"], test_size=0.2, random_state=42, stratify=processed["y"]
            )

            # Generate color palette for classes
            colors = self._generate_colors(processed["n_classes"])

            # Prepare data points for frontend - Convert NumPy types to Python native types
            train_points = []
            for i, (x, y_val) in enumerate(zip(X_train, y_train)):
                train_points.append({
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "class": int(y_val),  # Convert numpy.int64 to int
                    "class_name": str(processed["class_names"][int(y_val)]),  # Ensure index is int
                    "color": colors[int(y_val)],  # Ensure index is int
                    "type": "train",
                    "id": f"train_{i}"
                })

            test_points = []
            for i, (x, y_val) in enumerate(zip(X_test, y_test)):
                test_points.append({
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "class": int(y_val),  # Convert numpy.int64 to int
                    "class_name": str(processed["class_names"][int(y_val)]),  # Ensure index is int
                    "color": colors[int(y_val)],  # Ensure index is int
                    "type": "test",
                    "id": f"test_{i}"
                })

            # Dataset metadata - gerçek dosya ismini kullan
            actual_filename = self.resolve_filename(dataset_name)
            config = self.dataset_configs.get(actual_filename, {})

            return {
                "dataset_name": config.get("name", actual_filename),
                "description": config.get("description", ""),
                "method_used": method,
                "original_features": int(processed["n_features"]),  # Convert to int
                "n_samples": int(processed["n_samples"]),  # Convert to int
                "n_classes": int(processed["n_classes"]),  # Convert to int
                "class_names": processed["class_names"],  # Already converted above
                "feature_names": processed["feature_names"],  # Already a list of strings
                "train_points": train_points,
                "test_points": test_points,
                "all_points": train_points + test_points,
                "colors": colors,
                "bounds": {
                    "x_min": float(X_2d[:, 0].min()),
                    "x_max": float(X_2d[:, 0].max()),
                    "y_min": float(X_2d[:, 1].min()),
                    "y_max": float(X_2d[:, 1].max())
                }
            }

        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return {"error": str(e)}

    def _generate_colors(self, n_classes: int) -> List[str]:
        """Generate distinct colors for each class"""
        colors = [
            "#ef4444",  # Red
            "#22c55e",  # Green
            "#3b82f6",  # Blue
            "#f59e0b",  # Yellow
            "#8b5cf6",  # Purple
            "#ec4899",  # Pink
            "#06b6d4",  # Cyan
            "#84cc16",  # Lime
            "#f97316",  # Orange
            "#6366f1",  # Indigo
        ]

        if n_classes <= len(colors):
            return colors[:n_classes]

        # Generate more colors using HSL
        additional_colors = []
        for i in range(n_classes - len(colors)):
            hue = (i * 360 / (n_classes - len(colors))) % 360
            additional_colors.append(f"hsl({hue}, 70%, 50%)")

        return colors + additional_colors

    def get_available_datasets(self) -> List[Dict]:
        """Get list of available datasets"""
        available = []
        for filename, config in self.dataset_configs.items():
            file_path = os.path.join(self.datasets_path, filename)
            if os.path.exists(file_path):
                available.append({
                    "filename": filename,
                    "name": config.get("name", filename),
                    "description": config.get("description", ""),
                    "size": int(os.path.getsize(file_path))  # Convert to int
                })
        return available


# API endpoints (add these to your main FastAPI app)
def setup_dataset_routes(app):
    """Setup dataset visualization routes"""
    visualizer = DatasetVisualizer()

    @app.get("/api/datasets")
    async def get_available_datasets():
        """Get list of available datasets"""
        return {"datasets": visualizer.get_available_datasets()}

    @app.get("/api/datasets/{dataset_name}/visualize")
    async def visualize_dataset(dataset_name: str, method: str = "pca"):
        """Get visualization data for a specific dataset"""
        result = visualizer.generate_visualization_data(dataset_name, method)
        return result

    @app.get("/api/datasets/{dataset_name}/info")
    async def get_dataset_info(dataset_name: str):
        """Get basic information about a dataset"""
        df = visualizer.load_dataset(dataset_name)
        if df is None:
            return {"error": "Dataset not found"}

        # Gerçek dosya ismini al
        actual_filename = visualizer.resolve_filename(dataset_name)
        config = visualizer.dataset_configs.get(actual_filename, {})

        return {
            "name": config.get("name", actual_filename),
            "description": config.get("description", ""),
            "shape": [int(s) for s in df.shape],  # Convert to list of ints
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},  # Convert keys and values to strings
            "missing_values": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},  # Convert to int
            "sample_data": df.head().to_dict('records')
        }

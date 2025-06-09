# backend/app/cache_manager.py

import hashlib
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import fcntl
import time

class TrainingCacheManager:
    def __init__(self, cache_file_path: str = "/app/cache_data/training_cache.json"):
        self.cache_file_path = cache_file_path
        self.cache_data = {}

        # Cache dizinini oluştur
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

        self._load_cache()

    def _load_cache(self):
        """Cache dosyasını yükle"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    # File lock ile güvenli okuma
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    self.cache_data = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                print(f"Cache yüklendi: {len(self.cache_data)} kayıt")
            else:
                self.cache_data = {}
                print("Yeni cache dosyası oluşturulacak")
        except Exception as e:
            print(f"Cache yükleme hatası: {e}")
            self.cache_data = {}

    def _save_cache(self):
        """Cache'i dosyaya kaydet"""
        try:
            # Geçici dosyaya yaz, sonra atomic move
            temp_file = self.cache_file_path + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                # File lock ile güvenli yazma
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic move
            os.rename(temp_file, self.cache_file_path)

        except Exception as e:
            print(f"Cache kaydetme hatası: {e}")

    def generate_cache_key(
        self,
        algorithm: str,
        dataset_id: str,
        model_params: Dict[str, Any],
        global_settings: Dict[str, Any]
    ) -> str:
        """
        Benzersiz cache key oluştur
        """
        # selectedMetrics ve frontend_config_id'yi ignore et
        cleaned_params = {k: v for k, v in model_params.items()
                         if k not in ['selectedMetrics', 'frontend_config_id']}

        # Deterministic string oluştur
        cache_data = {
            "algorithm": algorithm,
            "dataset_id": dataset_id,
            "model_params": cleaned_params,
            "global_settings": global_settings
        }

        # JSON string'e çevir (sorted keys ile deterministic)
        cache_string = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))

        # SHA256 hash oluştur
        cache_key = hashlib.sha256(cache_string.encode()).hexdigest()

        return cache_key

    def get_cached_training_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Cache'den training sonucunu getir
        """
        try:
            if cache_key in self.cache_data:
                entry = self.cache_data[cache_key]

                # Hit count'u artır
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                entry["last_accessed"] = datetime.now().isoformat()

                print(f"🚀 Training Cache HIT: {cache_key[:8]}... (Hit count: {entry['hit_count']})")

                return {
                    "training_metrics": entry.get("training_metrics"),
                    "fit_time_seconds": entry.get("fit_time_seconds"),
                    "memory_usage_mb": entry.get("memory_usage_mb"),
                    "training_throughput": entry.get("training_throughput"),
                    "convergence_info": entry.get("convergence_info", {}),
                    "notes": entry.get("notes", [])
                }

            print(f"💻 Training Cache MISS: {cache_key[:8]}...")
            return None

        except Exception as e:
            print(f"Training cache okuma hatası: {e}")
            return None

    def save_training_result(
        self,
        cache_key: str,
        algorithm: str,
        dataset_id: str,
        model_params: Dict[str, Any],
        global_settings: Dict[str, Any],
        training_results: Dict[str, Any]
    ):
        """
        Training sonucunu cache'e kaydet
        """
        try:
            # selectedMetrics ve frontend_config_id'yi temizle
            cleaned_params = {k: v for k, v in model_params.items()
                             if k not in ['selectedMetrics', 'frontend_config_id']}

            cache_entry = {
                "algorithm": algorithm,
                "dataset_id": dataset_id,
                "model_params": cleaned_params,
                "global_settings": global_settings,

                # Training sonuçları
                "training_metrics": training_results.get("training_metrics"),
                "fit_time_seconds": training_results.get("fit_time_seconds"),
                "memory_usage_mb": training_results.get("memory_usage_mb"),
                "training_throughput": training_results.get("training_throughput"),
                "convergence_info": training_results.get("convergence_info", {}),
                "notes": training_results.get("notes", []),

                # Metadata
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "hit_count": 1,
                "type": "training"
            }

            # Memory'ye kaydet
            self.cache_data[cache_key] = cache_entry

            # Dosyaya kaydet
            self._save_cache()

            print(f"💾 Training cache'e kaydedildi: {cache_key[:8]}... (Toplam: {len(self.cache_data)} kayıt)")

        except Exception as e:
            print(f"Training cache kaydetme hatası: {e}")

    def get_cached_evaluation_result(self, cache_key: str, selected_metrics: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Cache'den evaluation sonucunu getir
        selected_metrics varsa sadece o metrikleri filtrele
        """
        try:
            eval_cache_key = f"eval_{cache_key}"
            if eval_cache_key in self.cache_data:
                entry = self.cache_data[eval_cache_key]

                # Hit count'u artır
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                entry["last_accessed"] = datetime.now().isoformat()

                print(f"🚀 Evaluation Cache HIT: {cache_key[:8]}... (Hit count: {entry['hit_count']})")

                # Cached metrics'i al
                cached_metrics = entry.get("metrics", {})

                # Eğer selected_metrics varsa filtrele
                if selected_metrics:
                    # Frontend'den gelen metrik adlarını backend format'ına çevir
                    metric_mapping = {
                        "Accuracy": "accuracy",
                        "Precision": "precision",
                        "Recall": "recall",
                        "F1-Score": "f1_score",
                        "ROC AUC": "roc_auc"
                    }

                    backend_selected = [metric_mapping.get(m, m.lower()) for m in selected_metrics]
                    filtered_metrics = {k: v for k, v in cached_metrics.items() if k in backend_selected}
                else:
                    filtered_metrics = cached_metrics

                return {
                    "metrics": filtered_metrics,
                    "plot_data": entry.get("plot_data", {}),
                    "score_time_seconds": entry.get("score_time_seconds"),
                    "prediction_performance": entry.get("prediction_performance", {}),
                    "notes": entry.get("notes", [])
                }

            print(f"💻 Evaluation Cache MISS: {cache_key[:8]}...")
            return None

        except Exception as e:
            print(f"Evaluation cache okuma hatası: {e}")
            return None

    def save_evaluation_result(
        self,
        cache_key: str,
        algorithm: str,
        dataset_id: str,
        model_params: Dict[str, Any],
        global_settings: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ):
        """
        Evaluation sonucunu cache'e kaydet (TÜM metrikleri kaydet)
        """
        try:
            # selectedMetrics ve frontend_config_id'yi temizle
            cleaned_params = {k: v for k, v in model_params.items()
                             if k not in ['selectedMetrics', 'frontend_config_id']}

            cache_entry = {
                "algorithm": algorithm,
                "dataset_id": dataset_id,
                "model_params": cleaned_params,
                "global_settings": global_settings,

                # Evaluation sonuçları - TÜM METRİKLERİ KAYDET
                "metrics": evaluation_results.get("metrics", {}),
                "plot_data": evaluation_results.get("plot_data", {}),
                "score_time_seconds": evaluation_results.get("score_time_seconds"),
                "prediction_performance": evaluation_results.get("prediction_performance", {}),
                "notes": evaluation_results.get("notes", []),

                # Metadata
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "hit_count": 1,
                "type": "evaluation"
            }

            # Memory'ye kaydet (eval_ prefix ile)
            eval_cache_key = f"eval_{cache_key}"
            self.cache_data[eval_cache_key] = cache_entry

            # Dosyaya kaydet
            self._save_cache()

            print(f"💾 Evaluation cache'e kaydedildi: {cache_key[:8]}... (Toplam: {len(self.cache_data)} kayıt)")

        except Exception as e:
            print(f"Evaluation cache kaydetme hatası: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Cache istatistiklerini getir
        """
        try:
            if not self.cache_data:
                return {
                    "total_entries": 0,
                    "cache_file_size_mb": 0,
                    "algorithms": [],
                    "datasets": []
                }

            algorithms = set()
            datasets = set()
            total_hits = 0
            training_entries = 0
            evaluation_entries = 0

            for key, entry in self.cache_data.items():
                algorithms.add(entry.get("algorithm", "unknown"))
                datasets.add(entry.get("dataset_id", "unknown"))
                total_hits += entry.get("hit_count", 0)

                if entry.get("type") == "training":
                    training_entries += 1
                elif entry.get("type") == "evaluation":
                    evaluation_entries += 1

            # Dosya boyutu
            file_size_mb = 0
            if os.path.exists(self.cache_file_path):
                file_size_mb = os.path.getsize(self.cache_file_path) / (1024 * 1024)

            return {
                "total_entries": len(self.cache_data),
                "training_entries": training_entries,
                "evaluation_entries": evaluation_entries,
                "total_hits": total_hits,
                "avg_hits_per_entry": total_hits / len(self.cache_data) if self.cache_data else 0,
                "cache_file_size_mb": round(file_size_mb, 2),
                "unique_algorithms": len(algorithms),
                "unique_datasets": len(datasets),
                "algorithms": list(algorithms),
                "datasets": list(datasets),
                "cache_file_path": self.cache_file_path
            }

        except Exception as e:
            print(f"Cache stats hatası: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Cache'i temizle"""
        try:
            self.cache_data = {}
            if os.path.exists(self.cache_file_path):
                os.remove(self.cache_file_path)
            print("Cache temizlendi")
        except Exception as e:
            print(f"Cache temizleme hatası: {e}")

# Global instance
cache_manager = TrainingCacheManager()

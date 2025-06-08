# backend/app/ml_models/decision_tree_classifier.py

import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List


"""
    Decision Tree modelini eğitir, değerlendirir ve sonuçları döndürür.
"""
def train_and_evaluate_dt(
    data_dict: Dict[str, Any],
    model_params_from_frontend: Dict[str, Any],
    global_settings: Dict[str, Any],
    mode: str = "evaluate"  # YENİ: "train" veya "evaluate"
) -> Dict[str, Any]:

    X_train_df = data_dict.get("X_train")
    y_train_series = data_dict.get("y_train")
    X_test_df = data_dict.get("X_test")
    y_test_series = data_dict.get("y_test")

    X_full_df = data_dict.get("X_full")
    y_full_series = data_dict.get("y_full")

    was_scaled_in_processor = data_dict.get("scaled", False)
    preparation_log = data_dict.get("data_preparation_log", [])
    results_log = list(preparation_log)

    raw_params = model_params_from_frontend.copy()
    selected_metrics_frontend_names = raw_params.pop('selectedMetrics', ['Accuracy'])

    # Frontend'den gelen frontend_config_id'yi alalım (sonuçlarda kullanmak için)
    frontend_config_id = raw_params.pop('frontend_config_id', None)

    dt_sklearn_params = {}
    dt_sklearn_params['criterion'] = raw_params.get('Criterion', 'gini').lower()
    max_depth_str = raw_params.get('Max Depth', '')
    dt_sklearn_params['max_depth'] = int(max_depth_str) if max_depth_str and max_depth_str.isdigit() else None
    min_samples_split_val = raw_params.get('Min Samples Split', 2)
    dt_sklearn_params['min_samples_split'] = int(min_samples_split_val) if min_samples_split_val is not None else 2
    min_samples_leaf_val = raw_params.get('Min Samples Leaf', 1)
    dt_sklearn_params['min_samples_leaf'] = int(min_samples_leaf_val) if min_samples_leaf_val is not None else 1

    random_state_val = global_settings.get('randomSeed')
    if random_state_val is not None:
        dt_sklearn_params['random_state'] = int(random_state_val)

    model_instance = DecisionTreeClassifier(**dt_sklearn_params)

    results_log.append(f"Decision Tree modeli Scikit-learn parametreleriyle oluşturuldu: {dt_sklearn_params}")
    print(f"Decision Tree modeli oluşturuldu. Parametreler: {dt_sklearn_params}")

    output_results = {
        "metrics": {} if mode == "evaluate" else None,  # Sadece evaluate için
        "training_metrics": {} if mode == "train" else None,  # Sadece train için
        "fit_time_seconds": 0.0,
        "score_time_seconds": 0.0,
        "memory_usage_mb": 0.0,  # YENİ: Bellek kullanımı
        "training_throughput": 0.0,  # YENİ: Eğitim hızı
        "convergence_info": {},  # YENİ: Yakınsama bilgisi
        "notes": results_log,
        "plot_data": {}
    }

    # --- Train/Test Split ile Eğitim ve Değerlendirme ---
    if global_settings.get('useTrainTestSplit') and X_train_df is not None and y_train_series is not None and X_test_df is not None and y_test_series is not None:
        results_log.append(f"Train/Test split ile {mode} yapılıyor.")

        # Bellek kullanımını ölç
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_fit_time = time.time()
        model_instance.fit(X_train_df, y_train_series)
        fit_duration = time.time() - start_fit_time
        output_results["fit_time_seconds"] = round(fit_duration, 4)

        # Bellek kullanımını hesapla
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        output_results["memory_usage_mb"] = round(memory_after - memory_before, 2)

        # Training throughput (örnek/saniye)
        total_samples = len(X_train_df)
        output_results["training_throughput"] = round(total_samples / fit_duration, 2)

        if mode == "train":
            # SADECE EĞİTİM PERFORMANSI METRİKLERİ
            output_results["training_metrics"] = {
                "fit_time_seconds": output_results["fit_time_seconds"],
                "memory_usage_mb": output_results["memory_usage_mb"],
                "training_throughput_samples_per_sec": output_results["training_throughput"],
                "training_samples_count": total_samples,
                "tree_depth": int(model_instance.get_depth()),  # int() ekle
                "tree_nodes": int(model_instance.tree_.node_count),  # int() ekle
                "tree_leaves": int(model_instance.get_n_leaves())  # int() ekle
            }
            # Decision Tree için ek bilgiler
            output_results["convergence_info"]["tree_structure"] = {
                "max_depth_reached": int(model_instance.get_depth()),  # int() ekle
                "total_nodes": int(model_instance.tree_.node_count),  # int() ekle
                "leaves_count": int(model_instance.get_n_leaves()),  # int() ekle
                "feature_importances": [float(x) for x in model_instance.feature_importances_[:5].tolist()] if len(model_instance.feature_importances_) > 0 else []  # float() ekle
            }

        elif mode == "evaluate":
            # SADECE TAHMİN BAŞARISI METRİKLERİ
            start_score_time = time.time()
            y_pred_test = model_instance.predict(X_test_df)
            y_pred_proba_test = None
            if "ROC AUC" in selected_metrics_frontend_names:
                try:
                    y_pred_proba_test = model_instance.predict_proba(X_test_df)
                except AttributeError:
                    results_log.append("ROC AUC için olasılık tahmini (predict_proba) alınamadı.")

            output_results["score_time_seconds"] = round(time.time() - start_score_time, 4)

            for metric_name in selected_metrics_frontend_names:
                try:
                    if metric_name == "Accuracy":
                        score = accuracy_score(y_test_series, y_pred_test)
                    elif metric_name == "Precision":
                        score = precision_score(y_test_series, y_pred_test, average='macro', zero_division=0)
                    elif metric_name == "Recall":
                        score = recall_score(y_test_series, y_pred_test, average='macro', zero_division=0)
                    elif metric_name == "F1-Score":
                        score = f1_score(y_test_series, y_pred_test, average='macro', zero_division=0)
                    elif metric_name == "ROC AUC":
                        if y_pred_proba_test is not None:
                            num_classes = len(np.unique(y_test_series))
                            if num_classes == 2:
                                score = roc_auc_score(y_test_series, y_pred_proba_test[:, 1])
                            else: # multiclass
                                score = roc_auc_score(y_test_series, y_pred_proba_test, average='weighted', multi_class='ovr')
                        else:
                            raise ValueError("Olasılık tahminleri yok.")
                    else:
                        score = None
                        results_log.append(f"Uyarı: '{metric_name}' metriği tanınmadı/hesaplanmadı (Train/Test).")

                    if score is not None:
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name not in output_results["metrics"]:
                        output_results["metrics"][metric_name] = "Hesaplanamadı"

                except Exception as e_metric:
                    print(f"Metrik hesaplama hatası ({metric_name}): {e_metric}")
                    output_results["metrics"][metric_name] = f"Hata: {type(e_metric).__name__}"

    # --- Cross-Validation ile Eğitim ve Değerlendirme ---
    elif global_settings.get('useCrossValidation') and X_full_df is not None and y_full_series is not None:
        results_log.append(f"Cross-validation ile {mode} yapılıyor.")
        cv_folds = global_settings.get('cvFolds', 5)

        if mode == "train":
            # Cross-validation için sadece eğitim metrikleri
            from sklearn.model_selection import cross_validate
            cv_results = cross_validate(
                model_instance, X_full_df, y_full_series,
                cv=cv_folds,
                scoring=['neg_mean_squared_error'],  # Dummy scoring, sadece fit time için
                return_train_score=False
            )

            avg_fit_time = np.mean(cv_results['fit_time'])
            total_samples = len(X_full_df)

            output_results["fit_time_seconds"] = round(avg_fit_time, 4)
            output_results["training_throughput"] = round(total_samples / avg_fit_time, 2)

            # Final model'i fit et detaylı bilgi için
            model_instance.fit(X_full_df, y_full_series)

            output_results["training_metrics"] = {
                "avg_fit_time_seconds": round(avg_fit_time, 4),
                "training_throughput_samples_per_sec": output_results["training_throughput"],
                "training_samples_count": total_samples,
                "cv_folds": cv_folds,
                "tree_depth": int(model_instance.get_depth()),  # int() ekle
                "tree_nodes": int(model_instance.tree_.node_count),  # int() ekle
                "tree_leaves": int(model_instance.get_n_leaves())  # int() ekle
            }

            output_results["convergence_info"]["tree_structure"] = {
                "max_depth_reached": int(model_instance.get_depth()),  # int() ekle
                "total_nodes": int(model_instance.tree_.node_count),  # int() ekle
                "leaves_count": int(model_instance.get_n_leaves()),  # int() ekle
                "feature_importances": [float(x) for x in model_instance.feature_importances_[:5].tolist()] if len(model_instance.feature_importances_) > 0 else []  # float() ekle
            }

        elif mode == "evaluate":
            # Cross-validation için başarı metrikleri
            from sklearn.model_selection import cross_validate
            start_cv_time = time.time()

            scoring_metrics = []
            for metric_name in selected_metrics_frontend_names:
                if metric_name == "Accuracy":
                    scoring_metrics.append('accuracy')
                elif metric_name == "Precision":
                    scoring_metrics.append('precision_macro')
                elif metric_name == "Recall":
                    scoring_metrics.append('recall_macro')
                elif metric_name == "F1-Score":
                    scoring_metrics.append('f1_macro')
                # ROC AUC için cross-validation karmaşık, şimdilik skip

            if scoring_metrics:
                cv_results = cross_validate(
                    model_instance, X_full_df, y_full_series,
                    cv=cv_folds,
                    scoring=scoring_metrics,
                    return_train_score=False
                )

                for metric_name in selected_metrics_frontend_names:
                    if metric_name == "Accuracy" and 'test_accuracy' in cv_results:
                        score = np.mean(cv_results['test_accuracy'])
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name == "Precision" and 'test_precision_macro' in cv_results:
                        score = np.mean(cv_results['test_precision_macro'])
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name == "Recall" and 'test_recall_macro' in cv_results:
                        score = np.mean(cv_results['test_recall_macro'])
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name == "F1-Score" and 'test_f1_macro' in cv_results:
                        score = np.mean(cv_results['test_f1_macro'])
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name == "ROC AUC":
                        output_results["metrics"][metric_name] = "CV ile hesaplanamadı"

            output_results["score_time_seconds"] = round(time.time() - start_cv_time, 4)

    else:
        err_msg = "Uygun eğitim yöntemi (Train/Test veya CV) seçilmedi veya gerekli veri (X_train/X_full) eksik."
        results_log.append(err_msg)
        if mode == "evaluate":
            output_results["metrics"]["Error"] = err_msg
        else:
            output_results["training_metrics"]["Error"] = err_msg

    if mode == "evaluate":
        print(f"Decision Tree Değerlendirme Sonuçları: {output_results['metrics']}")
    else:
        print(f"Decision Tree Eğitim Performans Sonuçları: {output_results['training_metrics']}")

    return output_results

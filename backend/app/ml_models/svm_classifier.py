# backend/app/ml_models/svm_classifier.py

import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List


"""
    SVM modelini eğitir, değerlendirir ve sonuçları döndürür.
"""
def train_and_evaluate_svm(
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

    # SVM parametrelerini frontend'den sklearn formatına çevir
    svm_sklearn_params = {}
    svm_sklearn_params['C'] = float(raw_params.get('C (Reg. Param)', 1.0))
    kernel_str = raw_params.get('Kernel', 'RBF').lower()
    svm_sklearn_params['kernel'] = kernel_str
    gamma_str = raw_params.get('Gamma', 'Scale').lower()
    svm_sklearn_params['gamma'] = gamma_str

    # Poly kernel için degree parametresi
    if kernel_str == 'poly':
        degree_val = raw_params.get('Degree (Poly Kernel)', 3)
        svm_sklearn_params['degree'] = int(degree_val) if degree_val is not None else 3

    # Probability estimation için (ROC AUC metriği için gerekli)
    if "ROC AUC" in selected_metrics_frontend_names:
        svm_sklearn_params['probability'] = True

    random_state_val = global_settings.get('randomSeed')
    if random_state_val is not None:
        svm_sklearn_params['random_state'] = int(random_state_val)

    model_instance = SVC(**svm_sklearn_params)

    results_log.append(f"SVM modeli Scikit-learn parametreleriyle oluşturuldu: {svm_sklearn_params}")
    print(f"SVM modeli oluşturuldu. Parametreler: {svm_sklearn_params}")

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

        # Timeout kontrolü için
        max_training_time = 300  # 5 dakika max

        try:
            model_instance.fit(X_train_df, y_train_series)
            fit_duration = time.time() - start_fit_time

            if fit_duration > max_training_time:
                results_log.append(f"Uyarı: Eğitim {max_training_time}s'yi aştı ({fit_duration:.2f}s)")

        except Exception as e:
            fit_duration = time.time() - start_fit_time
            results_log.append(f"Eğitim hatası: {str(e)}")
            raise e

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
                "model_complexity": len(getattr(model_instance, 'support_', [])) if hasattr(model_instance, 'support_') else 'N/A'  # SVM support vectors sayısı
            }
            # SVM için ek bilgiler
            if hasattr(model_instance, 'n_support_'):
                output_results["convergence_info"]["support_vectors_per_class"] = [int(x) for x in model_instance.n_support_.tolist()]  # int() ekle
            if hasattr(model_instance, 'n_iter_'):
                output_results["convergence_info"]["iterations"] = int(model_instance.n_iter_)  # int() ekle

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
        results_log.append("Cross-Validation ile eğitim ve değerlendirme yapılıyor.")
        cv_folds = int(global_settings.get('cvFolds', 5))

        # SVM için ölçeklendirme önemli, Pipeline kullanıyoruz
        scaler = None
        apply_scaling_setting = global_settings.get("applyFeatureScaling", True)
        scaler_type_setting = global_settings.get("scalerType", 'standard')

        if apply_scaling_setting and not was_scaled_in_processor:
            if scaler_type_setting == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            pipeline_for_cv = Pipeline(steps=[('scaler', scaler), ('model', model_instance)])
            results_log.append(f"CV için Pipeline kullanıldı: {scaler_type_setting}Scaler + SVM")
        else:
            pipeline_for_cv = Pipeline(steps=[('model', model_instance)])
            if was_scaled_in_processor:
                results_log.append("Veri zaten ölçeklenmiş, CV için sadece SVM modeli kullanılıyor.")

        # CV için kullanılacak scorer'ları hazırla
        cv_scoring_map = {}
        if "Accuracy" in selected_metrics_frontend_names: cv_scoring_map['accuracy_cv'] = 'accuracy'
        if "Precision" in selected_metrics_frontend_names: cv_scoring_map['precision_macro_cv'] = make_scorer(precision_score, average='macro', zero_division=0)
        if "Recall" in selected_metrics_frontend_names: cv_scoring_map['recall_macro_cv'] = make_scorer(recall_score, average='macro', zero_division=0)
        if "F1-Score" in selected_metrics_frontend_names: cv_scoring_map['f1_macro_cv'] = make_scorer(f1_score, average='macro', zero_division=0)
        if "ROC AUC" in selected_metrics_frontend_names:
            num_classes = len(np.unique(y_full_series))
            if num_classes == 2:
                cv_scoring_map['roc_auc_cv'] = 'roc_auc'
            else:
                cv_scoring_map['roc_auc_ovr_weighted_cv'] = make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True)

        if not cv_scoring_map:
            cv_scoring_map['accuracy_cv'] = 'accuracy'
            results_log.append("CV için aktif metrik bulunamadı, varsayılan olarak accuracy kullanılıyor.")

        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=svm_sklearn_params.get('random_state'))

        start_cv_time = time.time()
        cv_results_sklearn = cross_validate(
            pipeline_for_cv, X_full_df, y_full_series, cv=cv_strategy,
            scoring=cv_scoring_map, return_train_score=False, error_score='raise'
        )
        cv_duration = time.time() - start_cv_time

        output_results["fit_time_seconds"] = round(np.mean(cv_results_sklearn.get('fit_time', [0])), 4)
        output_results["score_time_seconds"] = round(np.mean(cv_results_sklearn.get('score_time', [0])), 4)
        results_log.append(f"{cv_folds}-fold CV ({', '.join(cv_scoring_map.keys())}) süresi: {cv_duration:.4f} saniye.")

        # Sonuçları frontend metrik adlarına göre kaydet
        metric_map_cv_to_fe = {
            'test_accuracy_cv': 'Accuracy',
            'test_precision_macro_cv': 'Precision',
            'test_recall_macro_cv': 'Recall',
            'test_f1_macro_cv': 'F1-Score',
            'test_roc_auc_cv': 'ROC AUC',
            'test_roc_auc_ovr_weighted_cv': 'ROC AUC'
        }

        for cv_key, fe_name in metric_map_cv_to_fe.items():
            if fe_name in selected_metrics_frontend_names and cv_key in cv_results_sklearn:
                output_results["metrics"][f"{fe_name} (CV Avg)"] = round(np.mean(cv_results_sklearn[cv_key]), 4)
                output_results["metrics"][f"{fe_name} (CV Std)"] = round(np.std(cv_results_sklearn[cv_key]), 4)
            elif fe_name in selected_metrics_frontend_names and cv_key not in cv_results_sklearn:
                 output_results["metrics"][f"{fe_name} (CV)"] = "Hesaplanamadı (CV Key Yok)"

    else:
        err_msg = "Uygun eğitim yöntemi (Train/Test veya CV) seçilmedi veya gerekli veri (X_train/X_full) eksik."
        results_log.append(err_msg)
        output_results["metrics"]["Error"] = err_msg

    print(f"SVM Değerlendirme Sonuçları: {output_results['metrics']}")
    return output_results

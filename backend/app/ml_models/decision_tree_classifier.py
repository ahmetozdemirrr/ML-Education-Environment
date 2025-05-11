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
    global_settings: Dict[str, Any]
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
        "metrics": {}, 
        "fit_time_seconds": 0.0, 
        "score_time_seconds": 0.0,
        "notes": results_log, 
        "plot_data": {}
    }

    # --- Train/Test Split ile Eğitim ve Değerlendirme ---
    if global_settings.get('useTrainTestSplit') and X_train_df is not None and y_train_series is not None and X_test_df is not None and y_test_series is not None:
        results_log.append("Train/Test split ile eğitim ve değerlendirme yapılıyor.")
        
        start_fit_time = time.time()
        model_instance.fit(X_train_df, y_train_series)
        output_results["fit_time_seconds"] = round(time.time() - start_fit_time, 4)

        start_score_time = time.time()
        y_pred_test = model_instance.predict(X_test_df)
        y_pred_proba_test = None
        if "ROC AUC" in selected_metrics_frontend_names: # Sadece ROC AUC için proba'ya ihtiyaç var
            try:
                y_pred_proba_test = model_instance.predict_proba(X_test_df)
            except AttributeError:
                results_log.append("ROC AUC için olasılık tahmini (predict_proba) alınamadı.")
        
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
                    score = None # Tanınmayan metrik
                    results_log.append(f"Uyarı: '{metric_name}' metriği tanınmadı/hesaplanmadı (Train/Test).")

                if score is not None:
                    output_results["metrics"][metric_name] = round(score, 4)
                elif metric_name not in output_results["metrics"]: # Hata oluşmadıysa ama score None ise
                    output_results["metrics"][metric_name] = "Hesaplanamadı"

            except Exception as e_metric:
                print(f"Metrik hesaplama hatası ({metric_name}): {e_metric}")
                output_results["metrics"][metric_name] = f"Hata: {type(e_metric).__name__}"
        
        output_results["score_time_seconds"] = round(time.time() - start_score_time, 4)

    # --- Cross-Validation ile Eğitim ve Değerlendirme ---
    elif global_settings.get('useCrossValidation') and X_full_df is not None and y_full_series is not None:
        results_log.append("Cross-Validation ile eğitim ve değerlendirme yapılıyor.")
        cv_folds = int(global_settings.get('cvFolds', 5))
        
        # Decision Tree ölçeklendirme gerektirmez. data_processor'dan gelen X_full_df ölçeklenmemiş olmalı.
        pipeline_for_cv = Pipeline(steps=[('model', model_instance)]) 
        
        # CV için kullanılacak scorer'ları hazırla
        cv_scoring_map = {}
        if "Accuracy" in selected_metrics_frontend_names: cv_scoring_map['accuracy_cv'] = 'accuracy' # sklearn string'i
        if "Precision" in selected_metrics_frontend_names: cv_scoring_map['precision_macro_cv'] = make_scorer(precision_score, average='macro', zero_division=0)
        if "Recall" in selected_metrics_frontend_names: cv_scoring_map['recall_macro_cv'] = make_scorer(recall_score, average='macro', zero_division=0)
        if "F1-Score" in selected_metrics_frontend_names: cv_scoring_map['f1_macro_cv'] = make_scorer(f1_score, average='macro', zero_division=0)
        if "ROC AUC" in selected_metrics_frontend_names:
            num_classes = len(np.unique(y_full_series))
            if num_classes == 2:
                cv_scoring_map['roc_auc_cv'] = 'roc_auc' # sklearn string'i, predict_proba kullanır
            else:
                cv_scoring_map['roc_auc_ovr_weighted_cv'] = make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True)
        
        if not cv_scoring_map: # Eğer hiç metrik seçilmemişse veya eşleşmemişse
            cv_scoring_map['accuracy_cv'] = 'accuracy' # Varsayılan olarak accuracy
            results_log.append("CV için aktif metrik bulunamadı, varsayılan olarak accuracy kullanılıyor.")

        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=dt_sklearn_params.get('random_state'))
        
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

    print(f"Decision Tree Değerlendirme Sonuçları: {output_results['metrics']}")
    return output_results
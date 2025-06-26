# backend/app/ml_models/ann_classifier.py

import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List


"""
    ANN (MLPClassifier) modelini eğitir, değerlendirir ve sonuçları döndürür.
"""
def train_and_evaluate_ann(
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

    # ANN parametrelerini frontend'den sklearn formatına çevir
    ann_sklearn_params = {}

    # Hidden layer sizes parsing (e.g., "100" -> (100,) or "50,20" -> (50, 20))
    hidden_layers_str = raw_params.get('Hidden Layer Sizes', '100').strip()
    try:
        if ',' in hidden_layers_str:
            hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())
        else:
            hidden_layers = (int(hidden_layers_str),)
        ann_sklearn_params['hidden_layer_sizes'] = hidden_layers
    except ValueError:
        ann_sklearn_params['hidden_layer_sizes'] = (100,)  # Varsayılan
        results_log.append(f"Hidden layer sizes parsing hatası, varsayılan (100,) kullanılıyor: {hidden_layers_str}")

    activation_str = raw_params.get('Activation', 'ReLU').lower()
    if activation_str == 'relu':
        ann_sklearn_params['activation'] = 'relu'
    elif activation_str == 'tanh':
        ann_sklearn_params['activation'] = 'tanh'
    elif activation_str == 'logistic':
        ann_sklearn_params['activation'] = 'logistic'
    elif activation_str == 'identity':
        ann_sklearn_params['activation'] = 'identity'
    else:
        ann_sklearn_params['activation'] = 'relu'  # Varsayılan

    solver_str = raw_params.get('Solver', 'Adam').lower()
    if solver_str == 'adam':
        ann_sklearn_params['solver'] = 'adam'
    elif solver_str == 'sgd':
        ann_sklearn_params['solver'] = 'sgd'
    elif solver_str == 'l-bfgs':
        ann_sklearn_params['solver'] = 'lbfgs'
    else:
        ann_sklearn_params['solver'] = 'adam'  # Varsayılan

    ann_sklearn_params['alpha'] = float(raw_params.get('Alpha (L2 Penalty)', 0.0001))

    # SGD için özel learning rate ayarı
    if ann_sklearn_params['solver'] == 'sgd':
        learning_rate_str = raw_params.get('Learning Rate (SGD)', 'Constant').lower()
        if learning_rate_str == 'constant':
            ann_sklearn_params['learning_rate'] = 'constant'
        elif learning_rate_str == 'invscaling':
            ann_sklearn_params['learning_rate'] = 'invscaling'
        elif learning_rate_str == 'adaptive':
            ann_sklearn_params['learning_rate'] = 'adaptive'
        else:
            ann_sklearn_params['learning_rate'] = 'constant'

    # Max iterations (convergence için)
    ann_sklearn_params['max_iter'] = 200  # Varsayılan, ANN için genellikle daha fazla iterasyon gerekir

    # Early stopping (büyük datasetler için yararlı)
    ann_sklearn_params['early_stopping'] = True
    ann_sklearn_params['validation_fraction'] = 0.1
    ann_sklearn_params['n_iter_no_change'] = 10

    random_state_val = global_settings.get('randomSeed')
    if random_state_val is not None:
        ann_sklearn_params['random_state'] = int(random_state_val)

    model_instance = MLPClassifier(**ann_sklearn_params)

    results_log.append(f"ANN (MLPClassifier) modeli Scikit-learn parametreleriyle oluşturuldu: {ann_sklearn_params}")
    print(f"ANN modeli oluşturuldu. Parametreler: {ann_sklearn_params}")

    # FIXED: Initialize output_results properly based on mode
    output_results = {
        "metrics": {} if mode == "evaluate" else None,  # Evaluation metrics
        "training_metrics": {} if mode == "train" else None,  # Training metrics
        "fit_time_seconds": 0.0,
        "score_time_seconds": 0.0,
        "memory_usage_mb": 0.0,
        "training_throughput": 0.0,
        "convergence_info": {},
        "notes": results_log,
        "plot_data": {}
    }

    # === TRAIN MODE: Focus on training performance ===
    if mode == "train":
        if global_settings.get('useTrainTestSplit') and X_train_df is not None and y_train_series is not None:
            results_log.append("Train mode: Measuring training performance with train/test split.")

            start_fit_time = time.time()
            training_samples_count = len(X_train_df)

            try:
                model_instance.fit(X_train_df, y_train_series)
                fit_duration = time.time() - start_fit_time

                # Training performance metrics
                output_results["fit_time_seconds"] = round(fit_duration, 4)
                output_results["training_throughput"] = round(training_samples_count / fit_duration, 2) if fit_duration > 0 else 0

                # Memory usage estimation (simplified)
                import sys
                model_memory_mb = sys.getsizeof(model_instance) / (1024 * 1024)
                output_results["memory_usage_mb"] = round(model_memory_mb, 2)

                # Training-specific metrics
                output_results["training_metrics"] = {
                    "fit_time_seconds": output_results["fit_time_seconds"],
                    "memory_usage_mb": output_results["memory_usage_mb"],
                    "training_throughput_samples_per_sec": output_results["training_throughput"],
                    "training_samples_count": training_samples_count,
                    "hidden_layer_sizes": ann_sklearn_params['hidden_layer_sizes'],
                    "activation": ann_sklearn_params['activation'],
                    "solver": ann_sklearn_params['solver'],
                    "alpha": ann_sklearn_params['alpha'],
                    "iterations": int(model_instance.n_iter_) if hasattr(model_instance, 'n_iter_') else 0
                }

                # Convergence info
                output_results["convergence_info"] = {
                    "iterations": int(model_instance.n_iter_) if hasattr(model_instance, 'n_iter_') else 0,
                    "converged": bool(model_instance.n_iter_ < ann_sklearn_params['max_iter']) if hasattr(model_instance, 'n_iter_') else True,
                    "loss": float(model_instance.loss_) if hasattr(model_instance, 'loss_') else None
                }

                if not output_results["convergence_info"]["converged"]:
                    results_log.append(f"Uyarı: Model {ann_sklearn_params['max_iter']} iterasyonda yakınsayamadı.")

            except Exception as e:
                results_log.append(f"Model training hatası: {e}")
                output_results["training_metrics"] = {"Error": f"Training hatası: {type(e).__name__}"}
                return output_results

        else:
            err_msg = "Train mode requires train/test split with valid training data."
            results_log.append(err_msg)
            output_results["training_metrics"] = {"Error": err_msg}

        print(f"ANN Training Performance Results: {output_results.get('training_metrics', {})}")
        return output_results

    # === EVALUATE MODE: Focus on prediction accuracy ===
    elif mode == "evaluate":
        # Train/Test Split ile Değerlendirme
        if global_settings.get('useTrainTestSplit') and X_train_df is not None and y_train_series is not None and X_test_df is not None and y_test_series is not None:
            results_log.append("Evaluate mode: Train/Test split ile eğitim ve değerlendirme yapılıyor.")

            start_fit_time = time.time()
            try:
                model_instance.fit(X_train_df, y_train_series)
                if hasattr(model_instance, 'n_iter_') and model_instance.n_iter_ >= ann_sklearn_params['max_iter']:
                    results_log.append(f"Uyarı: Model {ann_sklearn_params['max_iter']} iterasyonda yakınsayamadı.")
            except Exception as e:
                results_log.append(f"Model eğitim hatası: {e}")
                output_results["metrics"]["Error"] = f"Eğitim hatası: {type(e).__name__}"
                return output_results

            output_results["fit_time_seconds"] = round(time.time() - start_fit_time, 4)

            start_score_time = time.time()
            y_pred_test = model_instance.predict(X_test_df)
            y_pred_proba_test = None
            if "ROC AUC" in selected_metrics_frontend_names:  # Sadece ROC AUC için proba'ya ihtiyaç var
                try:
                    y_pred_proba_test = model_instance.predict_proba(X_test_df)
                except AttributeError:
                    results_log.append("ROC AUC için olasılık tahmini (predict_proba) alınamadı.")

            # Calculate each requested metric
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
                            else:  # multiclass
                                score = roc_auc_score(y_test_series, y_pred_proba_test, average='weighted', multi_class='ovr')
                        else:
                            raise ValueError("Olasılık tahminleri yok.")
                    else:
                        score = None  # Tanınmayan metrik
                        results_log.append(f"Uyarı: '{metric_name}' metriği tanınmadı/hesaplanmadı (Train/Test).")

                    if score is not None:
                        output_results["metrics"][metric_name] = round(score, 4)
                    elif metric_name not in output_results["metrics"]:  # Hata oluşmadıysa ama score None ise
                        output_results["metrics"][metric_name] = "Hesaplanamadı"

                except Exception as e_metric:
                    print(f"Metrik hesaplama hatası ({metric_name}): {e_metric}")
                    output_results["metrics"][metric_name] = f"Hata: {type(e_metric).__name__}"

            output_results["score_time_seconds"] = round(time.time() - start_score_time, 4)

        # Cross-Validation ile Değerlendirme
        elif global_settings.get('useCrossValidation') and X_full_df is not None and y_full_series is not None:
            results_log.append("Evaluate mode: Cross-Validation ile eğitim ve değerlendirme yapılıyor.")
            cv_folds = int(global_settings.get('cvFolds', 5))

            # ANN için ölçeklendirme kritik, Pipeline kullanıyoruz
            scaler = None
            apply_scaling_setting = global_settings.get("applyFeatureScaling", True)
            scaler_type_setting = global_settings.get("scalerType", 'standard')

            if apply_scaling_setting and not was_scaled_in_processor:
                if scaler_type_setting == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                pipeline_for_cv = Pipeline(steps=[('scaler', scaler), ('model', model_instance)])
                results_log.append(f"CV için Pipeline kullanıldı: {scaler_type_setting}Scaler + ANN")
            else:
                pipeline_for_cv = Pipeline(steps=[('model', model_instance)])
                if was_scaled_in_processor:
                    results_log.append("Veri zaten ölçeklenmiş, CV için sadece ANN modeli kullanılıyor.")

            # CV için kullanılacak scorer'ları hazırla
            cv_scoring_map = {}
            if "Accuracy" in selected_metrics_frontend_names: cv_scoring_map['accuracy_cv'] = 'accuracy'
            if "Precision" in selected_metrics_frontend_names: cv_scoring_map['precision_macro_cv'] = make_scorer(precision_score, average='macro', zero_division=0)
            if "Recall" in selected_metrics_frontend_names: cv_scoring_map['recall_macro_cv'] = make_scorer(recall_score, average='macro', zero_division=0)
            if "F1-Score" in selected_metrics_frontend_names: cv_scoring_map['f1_macro_cv'] = make_scorer(f1_score, average='macro', zero_division=0)
            if "ROC AUC" in selected_metrics_frontend_names:
                print("Uyarı: ROC AUC Cross Validation'da şu anda desteklenmiyor, atlanıyor.")

            if not cv_scoring_map:
                cv_scoring_map['accuracy_cv'] = 'accuracy'
                results_log.append("CV için aktif metrik bulunamadı, varsayılan olarak accuracy kullanılıyor.")

            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=ann_sklearn_params.get('random_state'))

            start_cv_time = time.time()
            try:
                cv_results_sklearn = cross_validate(
                    pipeline_for_cv, X_full_df, y_full_series, cv=cv_strategy,
                    scoring=cv_scoring_map, return_train_score=False, error_score='raise'
                )
            except Exception as e:
                results_log.append(f"Cross-validation hatası: {e}")
                output_results["metrics"]["Error"] = f"CV hatası: {type(e).__name__}"
                return output_results

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
            err_msg = "Evaluate mode requires either train/test split or cross-validation with valid data."
            results_log.append(err_msg)
            output_results["metrics"]["Error"] = err_msg

        print(f"ANN Evaluation Results: {output_results.get('metrics', {})}")
        return output_results

    else:
        # Invalid mode
        err_msg = f"Invalid mode: {mode}. Use 'train' or 'evaluate'."
        results_log.append(err_msg)
        if mode == "evaluate":
            output_results["metrics"]["Error"] = err_msg
        else:
            output_results["training_metrics"]["Error"] = err_msg
        return output_results

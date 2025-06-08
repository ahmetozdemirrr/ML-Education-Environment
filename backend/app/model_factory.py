# backend/app/model_factory.py

from typing import Dict, Any

from .ml_models import decision_tree_classifier
from .ml_models import logistic_regression
from .ml_models import svm_classifier
from .ml_models import knn_classifier
from .ml_models import ann_classifier

def run_model_pipeline(
    algorithm_name: str,
    model_params_from_frontend: Dict[str, Any],
    data_dict: Dict[str, Any],
    global_settings: Dict[str, Any],
    mode: str = "evaluate"  # YENİ: "train" veya "evaluate"
) -> Dict[str, Any]:

    print(f"Model Fabrikası: '{algorithm_name}' için {mode.upper()} modu işlem başlatılıyor...")
    results_log = data_dict.get("data_preparation_log", [])

    # Her ihtimale karşı, model_params_from_frontend'in bir kopyasıyla çalışalım
    current_model_params = model_params_from_frontend.copy()

    if algorithm_name == "Decision Tree":
        model_results = decision_tree_classifier.train_and_evaluate_dt(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode  # YENİ: mode parametresi geçir
        )
    elif algorithm_name == "Logistic Regression":
        model_results = logistic_regression.train_and_evaluate_lr(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode  # YENİ: mode parametresi geçir
        )
    elif algorithm_name == "SVM":
        model_results = svm_classifier.train_and_evaluate_svm(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode  # YENİ: mode parametresi geçir
        )
    elif algorithm_name == "K-Nearest Neighbor":
        model_results = knn_classifier.train_and_evaluate_knn(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode  # YENİ: mode parametresi geçir
        )
    elif algorithm_name == "Artificial Neural Network":
        model_results = ann_classifier.train_and_evaluate_ann(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode  # YENİ: mode parametresi geçir
        )
    else:
        error_message = f"Desteklenmeyen algoritma: {algorithm_name}"
        print(f"Hata: {error_message}")
        # Hata durumunda da bir sonuç sözlüğü döndürmek frontend için daha iyi olabilir
        return {
            "metrics": {"Error": error_message},
            "fit_time_seconds": 0.0,
            "score_time_seconds": 0.0,
            "notes": results_log + [error_message],
            "plot_data": {}
        }

    # Modelden gelen notları genel loglara ekleyelim (eğer varsa)
    if model_results.get("notes"):
        # data_processor'dan gelenler zaten model_results içinde olmalı (eğer dt modülü eklediyse)
        # Biz yine de birleştirelim, tekrarı önlemek için kontrol edilebilir.
        # Şimdilik model_results['notes'] yeterli.
        pass

    return model_results

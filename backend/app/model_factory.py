# backend/app/model_factory.py - Enhanced version

from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd

from .ml_models import decision_tree_classifier
from .ml_models import logistic_regression
from .ml_models import svm_classifier
from .ml_models import knn_classifier
from .ml_models import ann_classifier
from .model_results_collector import results_collector

def run_model_pipeline(
    algorithm_name: str,
    model_params_from_frontend: Dict[str, Any],
    data_dict: Dict[str, Any],
    global_settings: Dict[str, Any],
    mode: str = "evaluate"  # "train" veya "evaluate"
) -> Dict[str, Any]:

    print(f"Model Fabrikası: '{algorithm_name}' için {mode.upper()} modu işlem başlatılıyor...")
    results_log = data_dict.get("data_preparation_log", [])

    # Frontend config ID'sini al
    config_id = model_params_from_frontend.get("frontend_config_id", f"{algorithm_name}_{int(time.time())}")

    # Her ihtimale karşı, model_params_from_frontend'in bir kopyasıyla çalışalım
    current_model_params = model_params_from_frontend.copy()

    # Algorithm dispatcher
    if algorithm_name == "Decision Tree":
        model_results = decision_tree_classifier.train_and_evaluate_dt(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode
        )
    elif algorithm_name == "Logistic Regression":
        model_results = logistic_regression.train_and_evaluate_lr(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode
        )
    elif algorithm_name == "SVM":
        model_results = svm_classifier.train_and_evaluate_svm(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode
        )
    elif algorithm_name == "K-Nearest Neighbor":
        model_results = knn_classifier.train_and_evaluate_knn(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode
        )
    elif algorithm_name == "Artificial Neural Network":
        model_results = ann_classifier.train_and_evaluate_ann(
            data_dict=data_dict,
            model_params_from_frontend=current_model_params,
            global_settings=global_settings,
            mode=mode
        )
    else:
        error_message = f"Desteklenmeyen algoritma: {algorithm_name}"
        print(f"Hata: {error_message}")
        return {
            "metrics": {"Error": error_message} if mode == "evaluate" else None,
            "training_metrics": {"Error": error_message} if mode == "train" else None,
            "fit_time_seconds": 0.0,
            "score_time_seconds": 0.0,
            "notes": results_log + [error_message],
            "plot_data": {},
            "enhanced_results": {"error": error_message}
        }

    # Enhanced results collection
    enhanced_results = {}

    try:
        # Safe data extraction - DÜZELTME
        X_data = data_dict.get("X_full")
        if X_data is None:
            X_data = data_dict.get("X_train")

        # Güvenli boyut kontrolü
        if X_data is not None and hasattr(X_data, 'shape'):
            n_features = X_data.shape[1]
            n_samples = len(X_data)
        else:
            n_features = 0
            n_samples = 0

        enhanced_results["metadata"] = {
            "config_id": config_id,
            "algorithm": algorithm_name,
            "mode": mode,
            "dataset_info": {
                "n_features": n_features,
                "n_samples": n_samples,
                "scaled": data_dict.get("scaled", False)
            },
            "global_settings": global_settings,
            "model_params": {k: v for k, v in current_model_params.items() if k not in ['selectedMetrics', 'frontend_config_id']},
            "timestamp": time.time()
        }

        # Performance summary
        enhanced_results["performance_summary"] = {
            "execution_time": {
                "fit_time": model_results.get("fit_time_seconds", 0),
                "score_time": model_results.get("score_time_seconds", 0),
                "total_time": model_results.get("fit_time_seconds", 0) + model_results.get("score_time_seconds", 0)
            },
            "memory_usage": model_results.get("memory_usage_mb", 0),
            "throughput": model_results.get("training_throughput", 0)
        }

        # Results analysis based on mode
        if mode == "evaluate" and model_results.get("metrics"):
            enhanced_results["evaluation_analysis"] = _analyze_evaluation_results(
                model_results["metrics"], algorithm_name
            )
        elif mode == "train" and model_results.get("training_metrics"):
            enhanced_results["training_analysis"] = _analyze_training_results(
                model_results["training_metrics"], algorithm_name
            )

        # Convergence and model-specific insights
        if model_results.get("convergence_info"):
            enhanced_results["model_insights"] = _extract_model_insights(
                model_results["convergence_info"], algorithm_name
            )

        # Plot data enhancement
        if model_results.get("plot_data"):
            enhanced_results["visualization_data"] = model_results["plot_data"]
        else:
            enhanced_results["visualization_data"] = _generate_basic_plot_data(model_results, mode)

    except Exception as e:
        enhanced_results["collection_error"] = str(e)
        print(f"Enhanced results collection error: {e}")

    # Add enhanced results to the original results
    model_results["enhanced_results"] = enhanced_results

    # Add recommendations
    model_results["recommendations"] = _generate_recommendations(
        model_results, algorithm_name, mode, global_settings
    )

    return model_results

def _analyze_evaluation_results(metrics: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
    """Analyze evaluation results and provide insights"""
    analysis = {
        "performance_level": "Unknown",
        "metric_analysis": {},
        "strengths": [],
        "areas_for_improvement": []
    }

    try:
        # Extract numeric metrics
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics[key.lower()] = value

        # Overall performance assessment
        accuracy = numeric_metrics.get("accuracy", numeric_metrics.get("accuracy (cv avg)", 0))

        if accuracy >= 0.95:
            analysis["performance_level"] = "Excellent"
            analysis["strengths"].append("Very high accuracy achieved")
        elif accuracy >= 0.85:
            analysis["performance_level"] = "Good"
            analysis["strengths"].append("Good overall performance")
        elif accuracy >= 0.70:
            analysis["performance_level"] = "Fair"
            analysis["areas_for_improvement"].append("Accuracy could be improved")
        else:
            analysis["performance_level"] = "Poor"
            analysis["areas_for_improvement"].append("Significant improvement needed")

        # Metric-specific analysis
        precision = numeric_metrics.get("precision", numeric_metrics.get("precision (cv avg)", 0))
        recall = numeric_metrics.get("recall", numeric_metrics.get("recall (cv avg)", 0))
        f1_score = numeric_metrics.get("f1_score", numeric_metrics.get("f1-score (cv avg)", 0))

        analysis["metric_analysis"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "balanced_performance": abs(precision - recall) < 0.1 if precision > 0 and recall > 0 else False
        }

        # Algorithm-specific insights
        if algorithm_name == "Decision Tree":
            if accuracy > 0.9:
                analysis["areas_for_improvement"].append("Check for potential overfitting")
        elif algorithm_name == "SVM":
            if accuracy < 0.8:
                analysis["areas_for_improvement"].append("Consider feature scaling or different kernel")
        elif algorithm_name == "K-Nearest Neighbor":
            if accuracy < 0.8:
                analysis["areas_for_improvement"].append("Consider feature scaling and optimal K selection")

    except Exception as e:
        analysis["error"] = str(e)

    return analysis

def _analyze_training_results(training_metrics: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
    """Analyze training results and provide insights"""
    analysis = {
        "training_efficiency": "Unknown",
        "performance_characteristics": {},
        "optimization_suggestions": []
    }

    try:
        fit_time = training_metrics.get("fit_time_seconds", 0)
        memory_usage = training_metrics.get("memory_usage_mb", 0)
        throughput = training_metrics.get("training_throughput_samples_per_sec", 0)

        # Training efficiency assessment
        if fit_time < 1:
            analysis["training_efficiency"] = "Very Fast"
        elif fit_time < 10:
            analysis["training_efficiency"] = "Fast"
        elif fit_time < 60:
            analysis["training_efficiency"] = "Moderate"
        else:
            analysis["training_efficiency"] = "Slow"
            analysis["optimization_suggestions"].append("Consider feature reduction or model simplification")

        # Performance characteristics
        analysis["performance_characteristics"] = {
            "training_time": fit_time,
            "memory_footprint": memory_usage,
            "processing_speed": throughput,
            "efficiency_score": min(100, max(0, 100 - (fit_time * 10) - (memory_usage * 0.1)))
        }

        # Algorithm-specific suggestions
        if algorithm_name == "Decision Tree":
            tree_depth = training_metrics.get("tree_depth", 0)
            if tree_depth > 20:
                analysis["optimization_suggestions"].append("Consider limiting tree depth to prevent overfitting")
        elif algorithm_name == "Artificial Neural Network":
            iterations = training_metrics.get("iterations", 0)
            if iterations > 100:
                analysis["optimization_suggestions"].append("Model converged slowly - consider learning rate adjustment")

    except Exception as e:
        analysis["error"] = str(e)

    return analysis

def _extract_model_insights(convergence_info: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
    """Extract model-specific insights from convergence information"""
    insights = {
        "convergence_quality": "Unknown",
        "model_complexity": "Unknown",
        "stability_indicators": {}
    }

    try:
        if algorithm_name == "Decision Tree":
            depth = convergence_info.get("tree_structure", {}).get("max_depth_reached", 0)
            nodes = convergence_info.get("tree_structure", {}).get("total_nodes", 0)

            insights["model_complexity"] = "High" if depth > 15 or nodes > 1000 else "Moderate" if depth > 10 else "Low"
            insights["tree_structure"] = {
                "depth": depth,
                "nodes": nodes,
                "leaves": convergence_info.get("tree_structure", {}).get("leaves_count", 0)
            }

        elif algorithm_name == "SVM":
            support_vectors = convergence_info.get("support_vectors_per_class", [])
            total_sv = sum(support_vectors) if support_vectors else 0

            insights["model_complexity"] = "High" if total_sv > 1000 else "Moderate" if total_sv > 100 else "Low"
            insights["support_vector_info"] = {
                "total_support_vectors": total_sv,
                "per_class": support_vectors
            }

        elif algorithm_name == "Artificial Neural Network":
            iterations = convergence_info.get("iterations", 0)
            insights["convergence_quality"] = "Good" if iterations < 100 else "Slow"

    except Exception as e:
        insights["error"] = str(e)

    return insights

def _generate_basic_plot_data(model_results: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Generate basic plot data when not available"""
    plot_data = {
        "available_plots": [],
        "performance_metrics": {}
    }

    try:
        if mode == "evaluate" and model_results.get("metrics"):
            # Performance metrics chart data
            metrics = model_results["metrics"]
            plot_data["performance_metrics"] = {
                "labels": list(metrics.keys()),
                "values": [v for v in metrics.values() if isinstance(v, (int, float))]
            }
            plot_data["available_plots"].append("performance_metrics")

        elif mode == "train" and model_results.get("training_metrics"):
            # Training performance data
            training_metrics = model_results["training_metrics"]
            plot_data["training_performance"] = {
                "fit_time": training_metrics.get("fit_time_seconds", 0),
                "memory_usage": training_metrics.get("memory_usage_mb", 0),
                "throughput": training_metrics.get("training_throughput_samples_per_sec", 0)
            }
            plot_data["available_plots"].append("training_performance")

    except Exception as e:
        plot_data["error"] = str(e)

    return plot_data

def _generate_recommendations(
    model_results: Dict[str, Any],
    algorithm_name: str,
    mode: str,
    global_settings: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on results"""
    recommendations = []

    try:
        if mode == "evaluate":
            metrics = model_results.get("metrics", {})
            accuracy = None

            # Find accuracy metric
            for key, value in metrics.items():
                if "accuracy" in key.lower() and isinstance(value, (int, float)):
                    accuracy = value
                    break

            if accuracy is not None:
                if accuracy < 0.7:
                    recommendations.append("Consider collecting more training data")
                    recommendations.append("Try feature engineering or selection")

                if accuracy < 0.6:
                    recommendations.append("Current model may not be suitable for this dataset")
                    recommendations.append("Consider trying a different algorithm")

        elif mode == "train":
            training_metrics = model_results.get("training_metrics", {})
            fit_time = training_metrics.get("fit_time_seconds", 0)

            if fit_time > 60:
                recommendations.append("Training time is high - consider feature reduction")
                recommendations.append("Try simpler model configurations")

        # Global settings recommendations
        if not global_settings.get("applyFeatureScaling", True):
            if algorithm_name in ["SVM", "K-Nearest Neighbor", "Artificial Neural Network"]:
                recommendations.append("Consider enabling feature scaling for better performance")

        if global_settings.get("useTrainTestSplit", False):
            test_ratio = global_settings.get("testSplitRatio", 0.2)
            if test_ratio < 0.15:
                recommendations.append("Consider using a larger test set for more reliable evaluation")

    except Exception as e:
        recommendations.append(f"Error generating recommendations: {str(e)}")

    return recommendations

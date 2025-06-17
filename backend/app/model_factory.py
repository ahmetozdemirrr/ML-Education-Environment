# backend/app/model_factory.py - Enhanced version with proper ModelResultsCollector integration

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

    # FIXED: Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    model_results = convert_numpy_types(model_results)

    # NEW: Enhanced results collection using ModelResultsCollector
    enhanced_results = {}

    try:
        # Safe data extraction
        X_data = data_dict.get("X_full") or data_dict.get("X_train")
        y_data = data_dict.get("y_full") or data_dict.get("y_train")
        X_test = data_dict.get("X_test")
        y_test = data_dict.get("y_test")

        # Enhanced results metadata
        enhanced_results["metadata"] = {
            "config_id": config_id,
            "algorithm": algorithm_name,
            "mode": mode,
            "dataset_info": {
                "n_features": X_data.shape[1] if X_data is not None and hasattr(X_data, 'shape') else 0,
                "n_samples": len(X_data) if X_data is not None else 0,
                "scaled": data_dict.get("scaled", False)
            },
            "global_settings": global_settings,
            "model_params": {k: v for k, v in current_model_params.items() if k not in ['selectedMetrics', 'frontend_config_id']},
            "timestamp": time.time()
        }

        # FIXED: Use ModelResultsCollector for detailed analysis
        if mode == "evaluate" and X_test is not None and y_test is not None:
            # Create a dummy trained model for analysis
            # We need to recreate the model with same parameters and train it
            selected_metrics = current_model_params.get("selectedMetrics", [])

            # Get the model from the training results if available, or retrain quickly
            if algorithm_name == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier

                dt_sklearn_params = {}
                dt_sklearn_params['criterion'] = current_model_params.get('Criterion', 'gini').lower()
                max_depth_str = current_model_params.get('Max Depth', '')
                dt_sklearn_params['max_depth'] = int(max_depth_str) if max_depth_str and max_depth_str.isdigit() else None
                dt_sklearn_params['min_samples_split'] = int(current_model_params.get('Min Samples Split', 2))
                dt_sklearn_params['min_samples_leaf'] = int(current_model_params.get('Min Samples Leaf', 1))

                random_state_val = global_settings.get('randomSeed')
                if random_state_val is not None:
                    dt_sklearn_params['random_state'] = int(random_state_val)

                model_instance = DecisionTreeClassifier(**dt_sklearn_params)
                model_instance.fit(data_dict.get("X_train"), data_dict.get("y_train"))

            elif algorithm_name == "SVM":
                from sklearn.svm import SVC

                svm_sklearn_params = {}
                svm_sklearn_params['C'] = float(current_model_params.get('C (Reg. Param)', 1.0))
                svm_sklearn_params['kernel'] = current_model_params.get('Kernel', 'RBF').lower()
                svm_sklearn_params['gamma'] = current_model_params.get('Gamma', 'Scale').lower()
                svm_sklearn_params['probability'] = True  # For ROC AUC

                random_state_val = global_settings.get('randomSeed')
                if random_state_val is not None:
                    svm_sklearn_params['random_state'] = int(random_state_val)

                model_instance = SVC(**svm_sklearn_params)
                model_instance.fit(data_dict.get("X_train"), data_dict.get("y_train"))

            elif algorithm_name == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression

                lr_sklearn_params = {}
                penalty_str = current_model_params.get('Penalty', 'L2').lower()
                lr_sklearn_params['penalty'] = None if penalty_str == 'none' else penalty_str
                lr_sklearn_params['C'] = float(current_model_params.get('C (Reg. Strength)', 1.0))
                lr_sklearn_params['solver'] = current_model_params.get('Solver', 'lbfgs')
                lr_sklearn_params['max_iter'] = max(int(current_model_params.get('Max Iterations', 100)), 1000)

                random_state_val = global_settings.get('randomSeed')
                if random_state_val is not None:
                    lr_sklearn_params['random_state'] = int(random_state_val)

                model_instance = LogisticRegression(**lr_sklearn_params)
                model_instance.fit(data_dict.get("X_train"), data_dict.get("y_train"))

            else:
                model_instance = None
                print(f"Model recreation for {algorithm_name} not implemented for enhanced results")

            # Use ModelResultsCollector for detailed analysis
            if model_instance is not None:
                print(f"Generating enhanced results using ModelResultsCollector for {algorithm_name}")
                detailed_results = results_collector.collect_evaluation_results(
                    model=model_instance,
                    X_test=X_test,
                    y_test=y_test,
                    algorithm_name=algorithm_name,
                    config_id=config_id,
                    selected_metrics=selected_metrics
                )

                print(f"ModelResultsCollector generated: {list(detailed_results.keys())}")

                # Merge detailed results
                if detailed_results.get("plot_data"):
                    model_results["plot_data"] = detailed_results["plot_data"]
                    print(f"Plot data keys: {list(detailed_results['plot_data'].keys())}")

                if detailed_results.get("detailed_metrics"):
                    enhanced_results["detailed_classification_report"] = detailed_results["detailed_metrics"]

                if detailed_results.get("predictions"):
                    enhanced_results["prediction_analysis"] = detailed_results["predictions"]

                if detailed_results.get("performance_summary"):
                    enhanced_results["performance_summary"] = detailed_results["performance_summary"]

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

    except Exception as e:
        enhanced_results["collection_error"] = str(e)
        print(f"Enhanced results collection error: {e}")
        import traceback
        traceback.print_exc()

    # Add enhanced results to the original results
    model_results["enhanced_results"] = enhanced_results

    # Add recommendations
    model_results["recommendations"] = _generate_recommendations(
        model_results, algorithm_name, mode, global_settings
    )

    print(f"Final model_results keys: {list(model_results.keys())}")
    if model_results.get("plot_data"):
        print(f"Final plot_data keys: {list(model_results['plot_data'].keys())}")

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

# backend/app/model_results_collector.py

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, Any, List, Tuple, Optional
import json

class ModelResultsCollector:
    """
    Comprehensive model results collection and analysis
    """

    def __init__(self):
        self.results = {}

    def collect_training_results(
        self,
        model,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        algorithm_name="Unknown",
        config_id="unknown"
    ) -> Dict[str, Any]:
        """
        Collect comprehensive training results
        """
        start_time = time.time()

        results = {
            "config_id": config_id,
            "algorithm": algorithm_name,
            "timestamp": time.time(),
            "training_metrics": {},
            "model_characteristics": {},
            "performance_profile": {},
            "memory_usage": 0,
            "training_time": 0
        }

        try:
            # Training time
            fit_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - fit_start
            results["training_time"] = round(training_time, 4)

            # Model characteristics
            results["model_characteristics"] = self._extract_model_characteristics(
                model, algorithm_name, X_train, y_train
            )

            # Training performance
            if X_test is not None and y_test is not None:
                results["training_metrics"] = self._calculate_training_metrics(
                    model, X_train, y_train, X_test, y_test
                )

            # Memory usage estimation
            results["memory_usage"] = self._estimate_memory_usage(model, X_train)

        except Exception as e:
            results["error"] = str(e)

        results["collection_time"] = round(time.time() - start_time, 4)
        return results

    def collect_evaluation_results(
        self,
        model,
        X_test,
        y_test,
        algorithm_name="Unknown",
        config_id="unknown",
        selected_metrics=None
    ) -> Dict[str, Any]:
        """
        Collect comprehensive evaluation results
        """
        start_time = time.time()

        results = {
            "config_id": config_id,
            "algorithm": algorithm_name,
            "timestamp": time.time(),
            "metrics": {},
            "detailed_metrics": {},
            "plot_data": {},
            "predictions": {},
            "performance_summary": {}
        }

        try:
            # Predictions
            y_pred = model.predict(X_test)
            print(f"DEBUG - Predictions generated: {len(y_pred)} samples")

            # FIXED: Enhanced probability predictions with SVM support
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    print(f"DEBUG - predict_proba shape: {y_pred_proba.shape if y_pred_proba is not None else 'None'}")
                except Exception as e:
                    print(f"DEBUG - predict_proba failed: {e}")
            elif hasattr(model, 'decision_function'):
                try:
                    # SVM için decision_function'ı probability'ye çevir
                    decision_scores = model.decision_function(X_test)
                    print(f"DEBUG - Using decision_function for ROC AUC, shape: {decision_scores.shape}")

                    # Binary classification için decision function'ı [0,1] aralığına normalize et
                    if len(np.unique(y_test)) == 2:
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        if decision_scores.ndim == 1:
                            decision_scores = decision_scores.reshape(-1, 1)
                        normalized_scores = scaler.fit_transform(decision_scores).flatten()
                        # İki sınıf için proba benzeri format oluştur
                        y_pred_proba = np.column_stack([1-normalized_scores, normalized_scores])
                        print(f"DEBUG - Created pseudo-probabilities from decision function: {y_pred_proba.shape}")
                    else:
                        # Multi-class için decision function'ı softmax'a çevir
                        from scipy.special import softmax
                        y_pred_proba = softmax(decision_scores, axis=1)
                        print(f"DEBUG - Created multi-class probabilities using softmax: {y_pred_proba.shape}")
                except Exception as e:
                    print(f"DEBUG - decision_function conversion failed: {e}")
            else:
                print(f"DEBUG - No probability method available for {algorithm_name}")

            # Basic metrics
            results["metrics"] = self._calculate_basic_metrics(
                y_test, y_pred, y_pred_proba, selected_metrics
            )
            print(f"DEBUG - Calculated metrics: {list(results['metrics'].keys())}")

            # Detailed classification report
            results["detailed_metrics"] = self._generate_detailed_classification_report(
                y_test, y_pred
            )

            # FIXED: Plot data for visualizations with debug info
            results["plot_data"] = self._generate_plot_data(
                y_test, y_pred, y_pred_proba, algorithm_name
            )

            # DEBUG: Print plot data info
            print(f"DEBUG - Generated plot_data keys: {list(results['plot_data'].keys())}")
            if 'confusion_matrix' in results['plot_data']:
                cm_shape = results['plot_data']['confusion_matrix']['matrix']
                print(f"DEBUG - Confusion matrix shape: {len(cm_shape)}x{len(cm_shape[0]) if cm_shape else 0}")
                print(f"DEBUG - Confusion matrix labels: {results['plot_data']['confusion_matrix']['labels']}")

            # Prediction analysis
            results["predictions"] = self._analyze_predictions(
                y_test, y_pred, y_pred_proba
            )

            # Performance summary
            results["performance_summary"] = self._create_performance_summary(
                results["metrics"], results["detailed_metrics"]
            )

        except Exception as e:
            print(f"ERROR - collect_evaluation_results failed: {e}")
            results["error"] = str(e)

        results["collection_time"] = round(time.time() - start_time, 4)
        return results

    def _extract_model_characteristics(self, model, algorithm_name, X_train, y_train):
        """Extract model-specific characteristics"""
        characteristics = {
            "algorithm": algorithm_name,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            "n_classes": len(np.unique(y_train))
        }

        # Algorithm-specific characteristics
        if hasattr(model, 'get_depth'):  # Decision Tree
            characteristics["tree_depth"] = int(model.get_depth())
            characteristics["n_leaves"] = int(model.get_n_leaves())
            characteristics["n_nodes"] = int(model.tree_.node_count)

        elif hasattr(model, 'n_support_'):  # SVM
            characteristics["n_support_vectors"] = int(np.sum(model.n_support_))
            characteristics["support_vectors_per_class"] = [int(x) for x in model.n_support_]

        elif hasattr(model, 'n_neighbors'):  # KNN
            characteristics["k_neighbors"] = int(model.n_neighbors)
            characteristics["algorithm"] = str(model.algorithm)

        elif hasattr(model, 'coef_'):  # Linear models
            characteristics["n_coefficients"] = int(model.coef_.size)
            if hasattr(model, 'n_iter_'):
                if isinstance(model.n_iter_, np.ndarray):
                    characteristics["iterations"] = [int(x) for x in model.n_iter_]
                else:
                    characteristics["iterations"] = int(model.n_iter_)

        elif hasattr(model, 'n_layers_'):  # Neural Network
            characteristics["n_layers"] = int(model.n_layers_)
            characteristics["hidden_layer_sizes"] = [int(x) for x in model.hidden_layer_sizes]
            if hasattr(model, 'n_iter_'):
                characteristics["iterations"] = int(model.n_iter_)

        return characteristics

    def _calculate_training_metrics(self, model, X_train, y_train, X_test, y_test):
        """Calculate comprehensive training metrics"""
        metrics = {}

        # Training accuracy
        train_pred = model.predict(X_train)
        metrics["train_accuracy"] = round(accuracy_score(y_train, train_pred), 4)

        # Test accuracy
        test_pred = model.predict(X_test)
        metrics["test_accuracy"] = round(accuracy_score(y_test, test_pred), 4)

        # Overfitting detection
        metrics["accuracy_gap"] = round(metrics["train_accuracy"] - metrics["test_accuracy"], 4)
        metrics["overfitting_risk"] = "High" if metrics["accuracy_gap"] > 0.1 else "Low"

        # Prediction speed
        start_time = time.time()
        _ = model.predict(X_test[:min(100, len(X_test))])
        prediction_time = time.time() - start_time
        metrics["prediction_speed_samples_per_sec"] = round(min(100, len(X_test)) / prediction_time, 2)

        return metrics

    def _calculate_basic_metrics(self, y_true, y_pred, y_pred_proba=None, selected_metrics=None):
        """Calculate basic evaluation metrics"""
        metrics = {}

        if selected_metrics is None:
            selected_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

        print(f"DEBUG - Calculating metrics for: {selected_metrics}")

        # Calculate each metric
        for metric in selected_metrics:
            try:
                if metric == "Accuracy":
                    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
                    print(f"DEBUG - Accuracy: {metrics['accuracy']}")
                elif metric == "Precision":
                    metrics["precision"] = round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
                    print(f"DEBUG - Precision: {metrics['precision']}")
                elif metric == "Recall":
                    metrics["recall"] = round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4)
                    print(f"DEBUG - Recall: {metrics['recall']}")
                elif metric == "F1-Score":
                    metrics["f1_score"] = round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4)
                    print(f"DEBUG - F1-Score: {metrics['f1_score']}")
                elif metric == "ROC AUC":
                    # FIXED: Enhanced ROC AUC calculation with better debugging
                    try:
                        if y_pred_proba is not None:
                            unique_classes = len(np.unique(y_true))
                            print(f"DEBUG - ROC AUC calculation: {unique_classes} classes, proba shape: {y_pred_proba.shape}")

                            if unique_classes == 2:
                                metrics["roc_auc"] = round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
                                print(f"DEBUG - Binary ROC AUC: {metrics['roc_auc']}")
                            else:
                                metrics["roc_auc"] = round(roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr'), 4)
                                print(f"DEBUG - Multi-class ROC AUC: {metrics['roc_auc']}")
                        else:
                            print(f"DEBUG - ROC AUC skipped: no probability predictions available")
                            metrics["roc_auc"] = "N/A"
                    except Exception as e:
                        print(f"DEBUG - ROC AUC calculation failed: {e}")
                        metrics["roc_auc"] = f"Error: {str(e)}"

            except Exception as e:
                error_key = metric.lower().replace('-', '_').replace(' ', '_')
                metrics[error_key] = f"Error: {str(e)}"
                print(f"DEBUG - {metric} calculation failed: {e}")

        print(f"DEBUG - Final metrics: {metrics}")
        return metrics

    def _generate_detailed_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            return convert_types(report)
        except Exception as e:
            print(f"DEBUG - Classification report failed: {e}")
            return {"error": str(e)}

    def _generate_plot_data(self, y_true, y_pred, y_pred_proba, algorithm_name):
        """Generate data for various plots"""
        plot_data = {}

        try:
            # FIXED: Enhanced Confusion Matrix generation
            cm = confusion_matrix(y_true, y_pred)
            unique_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))

            plot_data["confusion_matrix"] = {
                "matrix": cm.tolist(),
                "labels": [str(label) for label in unique_labels]  # Ensure string labels
            }
            print(f"DEBUG - Confusion matrix created: {cm.shape} with labels: {unique_labels}")

            # Class distribution
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

            plot_data["class_distribution"] = {
                "true_distribution": dict(zip([str(x) for x in unique_true], [int(x) for x in counts_true])),
                "predicted_distribution": dict(zip([str(x) for x in unique_pred], [int(x) for x in counts_pred]))
            }

            # FIXED: Enhanced ROC Curve generation
            if y_pred_proba is not None:
                unique_classes = len(np.unique(y_true))
                print(f"DEBUG - Generating ROC curves for {unique_classes} classes")

                if unique_classes == 2:
                    # Binary classification
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
                    plot_data["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()
                    }
                    print(f"DEBUG - Binary ROC curve generated")

                    # Precision-Recall Curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
                    plot_data["precision_recall_curve"] = {
                        "precision": precision.tolist(),
                        "recall": recall.tolist(),
                        "thresholds": pr_thresholds.tolist()
                    }
                    print(f"DEBUG - Precision-Recall curve generated")

                # Prediction confidence distribution
                max_proba = np.max(y_pred_proba, axis=1)
                hist_counts, hist_bins = np.histogram(max_proba, bins=10)
                plot_data["confidence_distribution"] = {
                    "confidence_scores": max_proba.tolist(),
                    "histogram_counts": hist_counts.tolist(),
                    "histogram_bins": hist_bins.tolist()
                }
                print(f"DEBUG - Confidence distribution generated")

        except Exception as e:
            print(f"DEBUG - Plot data generation failed: {e}")
            plot_data["error"] = str(e)

        return plot_data

    def _analyze_predictions(self, y_true, y_pred, y_pred_proba):
        """Analyze prediction patterns"""
        analysis = {}

        try:
            # Correct vs incorrect predictions
            correct_mask = y_true == y_pred
            analysis["correct_predictions"] = int(np.sum(correct_mask))
            analysis["incorrect_predictions"] = int(np.sum(~correct_mask))
            analysis["accuracy_percentage"] = round(np.mean(correct_mask) * 100, 2)

            # Per-class performance
            classes = sorted(list(set(y_true.tolist())))
            per_class = {}
            for cls in classes:
                cls_mask = y_true == cls
                if np.any(cls_mask):
                    cls_correct = np.sum((y_true == y_pred) & cls_mask)
                    cls_total = np.sum(cls_mask)
                    per_class[f"class_{cls}"] = {
                        "total_samples": int(cls_total),
                        "correct_predictions": int(cls_correct),
                        "accuracy": round(cls_correct / cls_total, 4) if cls_total > 0 else 0
                    }
            analysis["per_class_performance"] = per_class

            # Confidence analysis (if available)
            if y_pred_proba is not None:
                max_confidence = np.max(y_pred_proba, axis=1)
                analysis["confidence_stats"] = {
                    "mean_confidence": round(np.mean(max_confidence), 4),
                    "median_confidence": round(np.median(max_confidence), 4),
                    "min_confidence": round(np.min(max_confidence), 4),
                    "max_confidence": round(np.max(max_confidence), 4)
                }

                # High/low confidence predictions
                high_conf_mask = max_confidence > 0.8
                low_conf_mask = max_confidence < 0.6

                analysis["confidence_analysis"] = {
                    "high_confidence_predictions": int(np.sum(high_conf_mask)),
                    "low_confidence_predictions": int(np.sum(low_conf_mask)),
                    "high_conf_accuracy": round(np.mean(correct_mask[high_conf_mask]), 4) if np.any(high_conf_mask) else 0,
                    "low_conf_accuracy": round(np.mean(correct_mask[low_conf_mask]), 4) if np.any(low_conf_mask) else 0
                }

        except Exception as e:
            print(f"DEBUG - Prediction analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def _create_performance_summary(self, metrics, detailed_metrics):
        """Create a performance summary"""
        summary = {
            "overall_performance": "Unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        try:
            accuracy = metrics.get("accuracy", 0)

            # Overall performance assessment
            if accuracy >= 0.9:
                summary["overall_performance"] = "Excellent"
            elif accuracy >= 0.8:
                summary["overall_performance"] = "Good"
            elif accuracy >= 0.7:
                summary["overall_performance"] = "Fair"
            else:
                summary["overall_performance"] = "Poor"

            # Identify strengths and weaknesses
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1 = metrics.get("f1_score", 0)

            if precision > 0.8:
                summary["strengths"].append("High precision - few false positives")
            if recall > 0.8:
                summary["strengths"].append("High recall - good at finding positive cases")
            if f1 > 0.8:
                summary["strengths"].append("Balanced precision and recall")

            if precision < 0.6:
                summary["weaknesses"].append("Low precision - many false positives")
                summary["recommendations"].append("Consider adjusting classification threshold")
            if recall < 0.6:
                summary["weaknesses"].append("Low recall - missing positive cases")
                summary["recommendations"].append("Consider feature engineering or more training data")

            # Class-specific analysis from detailed metrics
            if "macro avg" in detailed_metrics:
                macro_avg = detailed_metrics["macro avg"]
                if macro_avg.get("f1-score", 0) < 0.7:
                    summary["recommendations"].append("Consider class balancing techniques")

        except Exception as e:
            print(f"DEBUG - Performance summary failed: {e}")
            summary["error"] = str(e)

        return summary

    def _estimate_memory_usage(self, model, X):
        """Estimate model memory usage"""
        try:
            import sys

            # Base memory usage of the model object
            model_memory = sys.getsizeof(model)

            # Add memory for model-specific attributes
            if hasattr(model, 'tree_'):  # Decision Tree
                model_memory += sys.getsizeof(model.tree_)
            elif hasattr(model, 'support_vectors_'):  # SVM
                model_memory += sys.getsizeof(model.support_vectors_)
            elif hasattr(model, 'coef_'):  # Linear models
                model_memory += sys.getsizeof(model.coef_)

            # Convert to MB
            return round(model_memory / (1024 * 1024), 2)

        except Exception:
            return 0

# Global instance
results_collector = ModelResultsCollector()

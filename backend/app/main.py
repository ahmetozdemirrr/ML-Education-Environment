from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import time

from app import data_processor
from app import model_factory
from app.cache_manager import cache_manager

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "/app/mounted_datasets"

# Frontend'den gelecek isteklerin yapısını tanımlayan Pydantic modelleri
class SimulationRequest(BaseModel):
    algorithm: str
    params: Dict[str, Any]
    dataset: str
    global_settings: Dict[str, Any]

class DatasetInfo(BaseModel):
    id: str
    name: str

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında dataset dizininin varlığını ve içeriğini kontrol et."""
    print(f"Backend uygulaması başlatılıyor...")

    if not os.path.exists(DATA_DIR):
        print(f"UYARI: Dataset dizini '{DATA_DIR}' container içinde bulunamadı.")
        print("Lütfen docker-compose.yml dosyasındaki volume mount ayarlarını ve host makinenizdeki")
        print("'./project_datasets/datasets' klasörünün varlığını kontrol edin.")
    else:
        print(f"Datasetler şu dizinden sunulacak: '{DATA_DIR}'")
        try:
            if not os.listdir(DATA_DIR):
                print(f"UYARI: Dataset dizini '{DATA_DIR}' boş.")
                print("Lütfen dataset dosyalarınızı host makinenizdeki './project_datasets/datasets' klasörüne yerleştirin.")
            else:
                print(f"Bulunan dataset dosyaları (ilk 5): {os.listdir(DATA_DIR)[:5]}")
        except Exception as e:
            print(f"Dataset dizini okunurken hata: {e}")

@app.get("/datasets", response_model=List[DatasetInfo])
async def list_available_datasets():
    datasets = []
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=500,
                            detail=f"Kritik Hata: Dataset dizini '{DATA_DIR}' sunucuda bulunamadı.")

    try:
        for filename in os.listdir(DATA_DIR):
            if filename.startswith('.') or os.path.isdir(os.path.join(DATA_DIR, filename)):
                continue

            dataset_id = os.path.splitext(filename)[0]
            dataset_name = dataset_id.replace("_", " ").replace("-", " ").title()
            datasets.append(DatasetInfo(id=dataset_id, name=dataset_name))

    except Exception as e:
        print(f"'/datasets' endpoint'inde hata: Datasetler listelenirken sorun oluştu. {e}")
        raise HTTPException(status_code=500,
                            detail="Datasetler listelenirken bir sunucu hatası oluştu.")

    if not datasets:
        print(f"Uyarı: '/datasets' endpoint'i çağrıldı ancak '{DATA_DIR}' içinde uygun dosya bulunamadı.")

    return datasets

@app.post("/train")
async def process_training_request(request: Request, simulation_data: SimulationRequest):
    """
    SADECE EĞİTİM PERFORMANSI İÇİN:
    - Cache'den kontrol et, varsa direkt döndür
    - Yoksa hesapla ve cache'e kaydet
    - Enhanced results ile birlikte döndür
    """
    print("--- EĞİTİM PERFORMANSI İsteği Alındı ---")

    # Cache key oluştur
    cache_key = cache_manager.generate_cache_key(
        algorithm=simulation_data.algorithm,
        dataset_id=simulation_data.dataset,
        model_params=simulation_data.params,
        global_settings=simulation_data.global_settings
    )

    print(f"Cache Key: {cache_key[:16]}...")

    # Cache'den kontrol et
    cached_result = cache_manager.get_cached_evaluation_result(cache_key, selected_metrics)

    # VALIDATE CACHE DATA - boş metrics'leri reddet
    if cached_result and not cached_result.get("metrics"):
        print("⚠️ WARNING: Cache'de boş metrics bulundu, cache entry siliniyor...")
        cache_manager.delete_cache_entry(cache_key)
        cached_result = None

    if cached_result:
        print("🚀 Cache'den sonuç döndürülüyor!")
        unique_run_id = simulation_data.params.get("frontend_config_id", f"cached_{cache_key[:8]}")

        response_payload = create_enhanced_training_response(
            unique_run_id, simulation_data, cached_result, from_cache=True
        )
        return response_payload

    # Cache'de yok, hesapla
    print("💻 Cache'de yok, hesaplanıyor...")
    log_entry = {
        "algorithm": simulation_data.algorithm,
        "dataset_id": simulation_data.dataset,
        "params": simulation_data.params,
        "global_settings": simulation_data.global_settings,
    }
    print(f"İstek Detayları: {json.dumps(log_entry, indent=2)}")

    if not simulation_data.global_settings:
        raise HTTPException(status_code=400,
                            detail="İstek gövdesinde 'global_settings' alanı eksik.")

    try:
        # Veriyi hazırla
        prepared_data = data_processor.load_and_prepare_data(
            dataset_id=simulation_data.dataset,
            data_dir=DATA_DIR,
            global_settings=simulation_data.global_settings
        )

        # Model eğitimi - SADECE TRAIN MODE
        ml_results = model_factory.run_model_pipeline(
            algorithm_name=simulation_data.algorithm,
            model_params_from_frontend=simulation_data.params,
            data_dict=prepared_data,
            global_settings=simulation_data.global_settings,
            mode="train"
        )

        # Cache'e kaydet
        cache_manager.save_training_result(
            cache_key=cache_key,
            algorithm=simulation_data.algorithm,
            dataset_id=simulation_data.dataset,
            model_params=simulation_data.params,
            global_settings=simulation_data.global_settings,
            training_results=ml_results
        )

        unique_run_id = simulation_data.params.get("frontend_config_id", f"new_{cache_key[:8]}")

        response_payload = create_enhanced_training_response(
            unique_run_id, simulation_data, ml_results, from_cache=False
        )

        print(f"Training Performans Sonuçları: {response_payload['training_metrics']}")
        print("--- EĞİTİM İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except Exception as e:
        return handle_error(e, simulation_data)

@app.post("/evaluate")
async def process_evaluation_request(request: Request, simulation_data: SimulationRequest):
    """
    SADECE TAHMİN BAŞARISI İÇİN:
    - Cache'den kontrol et, varsa filtreleyip döndür
    - Yoksa hesapla ve cache'e kaydet (tüm metrikleri)
    - Enhanced results ile birlikte döndür
    """
    print("--- DEĞERLENDİRME (EVALUATE) İsteği Alındı ---")

    # Cache key oluştur (selectedMetrics olmadan)
    cache_key = cache_manager.generate_cache_key(
        algorithm=simulation_data.algorithm,
        dataset_id=simulation_data.dataset,
        model_params=simulation_data.params,
        global_settings=simulation_data.global_settings
    )

    print(f"Evaluation Cache Key: {cache_key[:16]}...")

    # Selected metrics'i al
    selected_metrics = simulation_data.params.get("selectedMetrics", [])

    # Cache'den kontrol et
    cached_result = cache_manager.get_cached_evaluation_result(cache_key, selected_metrics)

    if cached_result:
        print(f"🚀 Evaluation cache'den sonuç döndürülüyor! (Seçilen metrikler: {selected_metrics})")
        unique_run_id = simulation_data.params.get("frontend_config_id", f"eval_cached_{cache_key[:8]}")

        response_payload = create_enhanced_evaluation_response(
            unique_run_id, simulation_data, cached_result, from_cache=True
        )
        return response_payload

    # Cache'de yok, hesapla
    print("💻 Evaluation cache'de yok, hesaplanıyor...")
    log_entry = {
        "algorithm": simulation_data.algorithm,
        "dataset_id": simulation_data.dataset,
        "params": simulation_data.params,
        "global_settings": simulation_data.global_settings,
    }
    print(f"İstek Detayları: {json.dumps(log_entry, indent=2)}")

    if not simulation_data.global_settings:
        raise HTTPException(status_code=400,
                            detail="İstek gövdesinde 'global_settings' alanı eksik.")

    try:
        # Veriyi hazırla
        prepared_data = data_processor.load_and_prepare_data(
            dataset_id=simulation_data.dataset,
            data_dir=DATA_DIR,
            global_settings=simulation_data.global_settings
        )

        # Model değerlendirmesi - TÜM METRİKLERİ HESAPLA
        ml_results = model_factory.run_model_pipeline(
            algorithm_name=simulation_data.algorithm,
            model_params_from_frontend=simulation_data.params,
            data_dict=prepared_data,
            global_settings=simulation_data.global_settings,
            mode="evaluate"
        )
        # FIX: Debug the ml_results
        print(f"DEBUG - ML Results metrics before filtering: {ml_results.get('metrics', 'NO_METRICS')}")

        # Cache'e kaydet (TÜM metrikleri kaydet)
        cache_manager.save_evaluation_result(
            cache_key=cache_key,
            algorithm=simulation_data.algorithm,
            dataset_id=simulation_data.dataset,
            model_params=simulation_data.params,
            global_settings=simulation_data.global_settings,
            evaluation_results=ml_results
        )
        # FIX: Filtering logic - DON'T filter if no metrics exist
        selected_metrics = simulation_data.params.get("selectedMetrics", [])

        # Kullanıcının seçtiği metriklerle filtrelenmiş sonuç döndür
        if selected_metrics and ml_results.get("metrics") and len(ml_results["metrics"]) > 0:
            # Only filter if we actually have metrics to filter
            metric_mapping = {
                "Accuracy": ["accuracy", "Accuracy"],
                "Precision": ["precision", "Precision"],
                "Recall": ["recall", "Recall"],
                "F1-Score": ["f1_score", "F1-Score", "f1-score"],
                "ROC AUC": ["roc_auc", "ROC AUC"]
            }

            filtered_metrics = {}
            for selected_metric in selected_metrics:
                possible_keys = metric_mapping.get(selected_metric, [selected_metric.lower()])
                for key in possible_keys:
                    if key in ml_results["metrics"]:
                        filtered_metrics[key] = ml_results["metrics"][key]
                        break

            if filtered_metrics:  # Only apply filter if we found matching metrics
                ml_results["metrics"] = filtered_metrics
            # If no matches found, keep original metrics

        print(f"DEBUG - Final metrics to send: {ml_results.get('metrics', 'NO_METRICS')}")


        unique_run_id = simulation_data.params.get("frontend_config_id", f"eval_new_{cache_key[:8]}")

        response_payload = create_enhanced_evaluation_response(
            unique_run_id, simulation_data, ml_results, from_cache=False
        )

        # FIX: Better status checking
        if not ml_results.get("metrics") or len(ml_results.get("metrics", {})) == 0:
            response_payload["status"] = "warning"
            response_payload["overall_status_message"] = "Model değerlendirildi ancak metrikler hesaplanamadı."
            print("WARNING: No metrics calculated!")

        print(f"Evaluation Başarı Sonuçları: {response_payload['metrics']}")
        print("--- DEĞERLENDİRME İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except Exception as e:
        return handle_error(e, simulation_data)

def create_enhanced_training_response(unique_run_id, simulation_data, ml_results, from_cache=False):
    """Create enhanced training response with additional metadata and insights"""

    # FIXED: f-string içinde backslash kullanımı yerine ternary operator kullanıldı
    cache_message = "cache'den alındı (anında)" if from_cache else "başarıyla eğitildi ve cache'e kaydedildi"

    # Base response
    response_payload = {
        "configId": unique_run_id,
        "modelName": simulation_data.algorithm,
        "datasetId": simulation_data.dataset,
        "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
        "status": "success",
        "training_metrics": ml_results.get("training_metrics", {}),
        "fit_time_seconds": ml_results.get("fit_time_seconds"),
        "memory_usage_mb": ml_results.get("memory_usage_mb"),
        "training_throughput": ml_results.get("training_throughput"),
        "convergence_info": ml_results.get("convergence_info", {}),
        "notes_from_model": ml_results.get("notes", []),
        "overall_status_message": f"Model sonuçları {cache_message}.",
        "from_cache": from_cache
    }

    # Add enhanced results if available
    if ml_results.get("enhanced_results"):
        response_payload["enhanced_results"] = ml_results["enhanced_results"]

    # Add recommendations if available
    if ml_results.get("recommendations"):
        response_payload["recommendations"] = ml_results["recommendations"]

    # Add plot data if available
    if ml_results.get("plot_data"):
        response_payload["plotData"] = ml_results["plot_data"]
        print(f"DEBUG - PlotData keys added to response: {list(ml_results['plot_data'].keys())}")

    # Add execution metadata
    response_payload["execution_metadata"] = {
        "timestamp": time.time(),
        "cache_hit": from_cache,
        "algorithm": simulation_data.algorithm,
        "dataset": simulation_data.dataset,
        "mode": "training"
    }

    return response_payload

def create_enhanced_evaluation_response(unique_run_id, simulation_data, ml_results, from_cache=False):
    """Create enhanced evaluation response with additional metadata and insights"""

    # FIXED: f-string içinde backslash kullanımı yerine ternary operator kullanıldı
    cache_message = "cache'den alındı (anında)" if from_cache else "başarıyla değerlendirildi ve cache'e kaydedildi"

    # FIX: Metrics processing
    metrics = ml_results.get("metrics", {})

    # Ensure metrics are properly formatted and not filtered incorrectly
    if not metrics and ml_results.get("enhanced_results", {}).get("metadata", {}).get("mode") == "evaluate":
        # Try to get metrics from enhanced_results if main metrics is empty
        metrics = ml_results.get("enhanced_results", {}).get("evaluation_metrics", {})

    # Base response
    response_payload = {
        "configId": unique_run_id,
        "modelName": simulation_data.algorithm,
        "datasetId": simulation_data.dataset,
        "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
        "status": "success",
        "metrics": metrics,  # FIX: Make sure this is not empty
        "plotData": ml_results.get("plot_data", {}),
        "score_time_seconds": ml_results.get("score_time_seconds"),
        "prediction_performance": ml_results.get("prediction_performance", {}),
        "notes_from_model": ml_results.get("notes", []),
        "overall_status_message": f"Evaluation sonuçları {cache_message}.",
        "from_cache": from_cache
    }

    # Add enhanced results if available
    if ml_results.get("enhanced_results"):
        response_payload["enhanced_results"] = ml_results["enhanced_results"]

    # Add recommendations if available
    if ml_results.get("recommendations"):
        response_payload["recommendations"] = ml_results["recommendations"]

    # Add detailed metrics if available
    if ml_results.get("detailed_metrics"):
        response_payload["detailed_metrics"] = ml_results["detailed_metrics"]

    # FIX: Debug logging
    print(f"DEBUG - Response metrics: {response_payload['metrics']}")
    print(f"DEBUG - Original ml_results metrics: {ml_results.get('metrics', 'NO_METRICS')}")

    # Add execution metadata
    response_payload["execution_metadata"] = {
        "timestamp": time.time(),
        "cache_hit": from_cache,
        "algorithm": simulation_data.algorithm,
        "dataset": simulation_data.dataset,
        "mode": "evaluation",
        "selected_metrics": simulation_data.params.get("selectedMetrics", [])
    }

    return response_payload

@app.get("/cache-stats")
async def get_cache_statistics():
    """Cache istatistiklerini döndür"""
    stats = cache_manager.get_cache_stats()
    return {"cache_stats": stats}

@app.delete("/clear-cache")
async def clear_training_cache():
    """Cache'i temizle"""
    cache_manager.clear_cache()
    return {"message": "Training cache temizlendi"}

@app.get("/results-summary")
async def get_results_summary():
    """
    Sistemdeki tüm sonuçların özetini döndür
    Bu endpoint frontend'in analiz yapması için kullanılabilir
    """
    try:
        stats = cache_manager.get_cache_stats()

        summary = {
            "total_cached_results": stats.get("total_entries", 0),
            "training_results": stats.get("training_entries", 0),
            "evaluation_results": stats.get("evaluation_entries", 0),
            "unique_algorithms": stats.get("unique_algorithms", 0),
            "unique_datasets": stats.get("unique_datasets", 0),
            "algorithms": stats.get("algorithms", []),
            "datasets": stats.get("datasets", []),
            "cache_hit_rate": stats.get("avg_hits_per_entry", 0),
            "cache_size_mb": stats.get("cache_file_size_mb", 0)
        }

        return {"results_summary": summary}

    except Exception as e:
        return {"error": f"Results summary oluşturulurken hata: {str(e)}"}

@app.get("/model-insights/{algorithm_name}")
async def get_model_insights(algorithm_name: str):
    """
    Belirli bir algoritma için insights ve öneriler döndür
    """
    insights = {
        "algorithm": algorithm_name,
        "strengths": [],
        "weaknesses": [],
        "best_use_cases": [],
        "parameter_tips": [],
        "performance_expectations": {}
    }

    # Algorithm-specific insights
    if algorithm_name == "Decision Tree":
        insights.update({
            "strengths": [
                "Highly interpretable and explainable",
                "Handles both numerical and categorical features",
                "No need for feature scaling",
                "Can capture non-linear relationships"
            ],
            "weaknesses": [
                "Prone to overfitting, especially with deep trees",
                "Can be unstable (small changes in data can result in different trees)",
                "Biased toward features with more levels"
            ],
            "best_use_cases": [
                "When model interpretability is crucial",
                "Mixed data types (numerical and categorical)",
                "Rule-based decision making scenarios"
            ],
            "parameter_tips": [
                "Limit max_depth to prevent overfitting (try 3-10 for small datasets)",
                "Increase min_samples_split for complex datasets",
                "Use Gini for speed, Entropy for slightly better accuracy"
            ]
        })
    elif algorithm_name == "SVM":
        insights.update({
            "strengths": [
                "Effective in high-dimensional spaces",
                "Memory efficient (uses support vectors)",
                "Versatile (different kernel functions)",
                "Works well with clear margin separation"
            ],
            "weaknesses": [
                "Slow on large datasets",
                "Sensitive to feature scaling",
                "No probabilistic output (unless probability=True)",
                "Choice of kernel and parameters can be tricky"
            ],
            "best_use_cases": [
                "Text classification and high-dimensional data",
                "Small to medium datasets",
                "When clear decision boundaries exist"
            ],
            "parameter_tips": [
                "Always apply feature scaling",
                "Start with RBF kernel for non-linear problems",
                "Use Grid Search for C and gamma parameters",
                "Consider Linear kernel for large datasets"
            ]
        })
    # Add more algorithms as needed...

    return {"model_insights": insights}

def handle_error(e, simulation_data):
    """Ortak hata yönetimi fonksiyonu"""
    import traceback

    if isinstance(e, FileNotFoundError):
        print(f"Dosya Bulunamadı Hatası: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, ValueError):
        print(f"Değer/Veri Hatası: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    else:
        print(f"Beklenmedik Genel Hata ('{simulation_data.dataset}', Algoritma: '{simulation_data.algorithm}') işlenirken:")
        traceback.print_exc()

        error_detail_for_frontend = f"Sunucuda beklenmedik bir hata oluştu: {type(e).__name__}"

        return JSONResponse(
            status_code=500,
            content={
                "configId": simulation_data.params.get("frontend_config_id", f"{simulation_data.algorithm}_{simulation_data.dataset}_error"),
                "modelName": simulation_data.algorithm,
                "datasetId": simulation_data.dataset,
                "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
                "status": "error",
                "errorMessage": error_detail_for_frontend,
                "notes_from_model": [str(e)],
                "execution_metadata": {
                    "timestamp": time.time(),
                    "cache_hit": False,
                    "algorithm": simulation_data.algorithm,
                    "dataset": simulation_data.dataset,
                    "error": True
                }
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "Backend is running"}

@app.get("/system-info")
async def system_info():
    """Sistem bilgilerini döndür"""
    import psutil
    import platform

    try:
        return {
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            "cache": cache_manager.get_cache_stats(),
            "datasets_available": len([f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.json'))]) if os.path.exists(DATA_DIR) else 0
        }
    except Exception as e:
        return {"error": f"System info alınırken hata: {str(e)}"}

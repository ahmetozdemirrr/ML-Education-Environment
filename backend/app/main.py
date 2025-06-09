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

app = FastAPI() # fastapi objesi oluştur

# CORS settings - DÜZELTİLDİ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver
    allow_credentials=False,  # Credentials kapatıldı
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
    Docker volume ile mount edilen datasetlerin container içindeki yolu.
    Bu yol, docker-compose.yml dosyasında belirtilmiştir, konteyner ile
    host makine arasında bir bağlantı sağlar
"""
DATA_DIR = "/app/mounted_datasets"

# Frontend'den gelecek isteklerin yapısını tanımlayan Pydantic modelleri

# /train endpointine gelen POST isteklerinin yapısı:
class SimulationRequest(BaseModel):
    algorithm: str
    params: Dict[str, Any] # Model parametreleri ve seçilen metrikleri içerecek
    dataset: str           # Dataset ID'si (dosya adının uzantısız hali)
    global_settings: Dict[str, Any] # Frontend'deki GlobalSettingsPanel'den gelen ayarlar

# /datasets enpointinin döndüreceği veri yapısı:
class DatasetInfo(BaseModel):
    id: str
    name: str


@app.on_event("startup") # FastAPI başladığında ele alınan rutin
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


"""
    /datasets endpointine gelen GET isteklerini ele alır data set listesi
    ([dataset_id, dataset_isim] gibi) döner, boş bir liste oluşturur DATA_DIR'ı
    kontrol eder, klasör yoksa hata döner HTTPException ile
"""
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
    cached_result = cache_manager.get_cached_training_result(cache_key)

    if cached_result:
        print("🚀 Cache'den sonuç döndürülüyor!")
        unique_run_id = simulation_data.params.get("frontend_config_id", f"cached_{cache_key[:8]}")

        response_payload = {
            "configId": unique_run_id,
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success",
            "training_metrics": cached_result["training_metrics"],
            "fit_time_seconds": cached_result["fit_time_seconds"],
            "memory_usage_mb": cached_result["memory_usage_mb"],
            "training_throughput": cached_result["training_throughput"],
            "convergence_info": cached_result["convergence_info"],
            "notes_from_model": cached_result["notes"],
            "overall_status_message": "Model sonuçları cache'den alındı (anında).",
            "from_cache": True
        }

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
            "overall_status_message": "Model başarıyla eğitildi ve cache'e kaydedildi.",
            "from_cache": False
        }

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

        response_payload = {
            "configId": unique_run_id,
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success",
            "metrics": cached_result["metrics"],  # Filtrelenmiş metrikler
            "plotData": cached_result["plot_data"],
            "score_time_seconds": cached_result["score_time_seconds"],
            "prediction_performance": cached_result["prediction_performance"],
            "notes_from_model": cached_result["notes"],
            "overall_status_message": "Evaluation sonuçları cache'den alındı (anında).",
            "from_cache": True
        }

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

        # Cache'e kaydet (TÜM metrikleri kaydet)
        cache_manager.save_evaluation_result(
            cache_key=cache_key,
            algorithm=simulation_data.algorithm,
            dataset_id=simulation_data.dataset,
            model_params=simulation_data.params,
            global_settings=simulation_data.global_settings,
            evaluation_results=ml_results
        )

        # Kullanıcının seçtiği metriklerle filtrelenmiş sonuç döndür
        if selected_metrics and ml_results.get("metrics"):
            metric_mapping = {
                "Accuracy": "accuracy",
                "Precision": "precision",
                "Recall": "recall",
                "F1-Score": "f1_score",
                "ROC AUC": "roc_auc"
            }
            backend_selected = [metric_mapping.get(m, m.lower()) for m in selected_metrics]
            filtered_metrics = {k: v for k, v in ml_results["metrics"].items() if k in backend_selected}
            ml_results["metrics"] = filtered_metrics

        unique_run_id = simulation_data.params.get("frontend_config_id", f"eval_new_{cache_key[:8]}")

        response_payload = {
            "configId": unique_run_id,
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success",
            "metrics": ml_results.get("metrics", {}),  # Filtrelenmiş metrikler
            "plotData": ml_results.get("plot_data", {}),
            "score_time_seconds": ml_results.get("score_time_seconds"),
            "prediction_performance": ml_results.get("prediction_performance", {}),
            "notes_from_model": ml_results.get("notes", []),
            "overall_status_message": "Model başarıyla değerlendirildi ve cache'e kaydedildi.",
            "from_cache": False
        }

        if not ml_results.get("metrics"):
            response_payload["status"] = "warning"
            response_payload["overall_status_message"] = "Model değerlendirildi ancak metrikler hesaplanamadı."

        print(f"Evaluation Başarı Sonuçları: {response_payload['metrics']}")
        print("--- DEĞERLENDİRME İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except Exception as e:
        return handle_error(e, simulation_data)


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
            }
        )

"""
    /health endpoint is a standard endpoint that is usually used to check
    if an application is healthy (running)

    - can be tested with `curl -X GET "http://localhost:8000/health"`
    - or with Python: response = requests.get("http://localhost:8000/health")
"""
@app.get("/health")
async def health_check():
    return {"status": "Backend is running"}

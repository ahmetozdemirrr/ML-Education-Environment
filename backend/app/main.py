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


"""
/train ve /evaluate endpointlerini düzeltelim - main.py'deki değişiklikler
"""

@app.post("/train")
async def process_training_request(request: Request, simulation_data: SimulationRequest):
    """
    SADECE EĞİTİM PERFORMANSI İÇİN:
    - Fit time (eğitim süresi)
    - Memory usage
    - Training throughput
    - Convergence info
    """
    print("--- EĞİTİM PERFORMANSI İsteği Alındı ---")
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
        # load, preprocess and split data using the data_processor module
        prepared_data = data_processor.load_and_prepare_data(
            dataset_id=simulation_data.dataset,
            data_dir=DATA_DIR,
            global_settings=simulation_data.global_settings
        )

        # Model eğitimi - SADECE HIZLI METRIKLER
        ml_results = model_factory.run_model_pipeline(
            algorithm_name=simulation_data.algorithm,
            model_params_from_frontend=simulation_data.params,
            data_dict=prepared_data,
            global_settings=simulation_data.global_settings,
            mode="train"  # YENİ: mode parametresi
        )

        unique_run_id = f"{simulation_data.algorithm}_{simulation_data.dataset}_{str(time.time()).replace('.', '')}"

        # TRAIN response - sadece hız metrikleri
        response_payload = {
            "configId": simulation_data.params.get("frontend_config_id", unique_run_id),
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success",
            "training_metrics": ml_results.get("training_metrics", {}),  # Hız metrikleri
            "fit_time_seconds": ml_results.get("fit_time_seconds"),
            "memory_usage_mb": ml_results.get("memory_usage_mb"),
            "training_throughput": ml_results.get("training_throughput"),
            "convergence_info": ml_results.get("convergence_info", {}),
            "notes_from_model": ml_results.get("notes", []),
            "overall_status_message": "Model başarıyla eğitildi. Performans metrikleri hesaplandı."
        }

        print(f"Training Performans Sonuçları: {response_payload['training_metrics']}")
        print("--- EĞİTİM İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except Exception as e:
        # Hata yönetimi aynı...
        return handle_error(e, simulation_data)


@app.post("/evaluate")
async def process_evaluation_request(request: Request, simulation_data: SimulationRequest):
    """
    SADECE TAHMİN BAŞARISI İÇİN:
    - Accuracy, Precision, Recall, F1-Score, ROC AUC
    - Score time (tahmin süresi)
    - Prediction performance
    """
    print("--- DEĞERLENDİRME (EVALUATE) İsteği Alındı ---")
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
        # load, preprocess and split data using the data_processor module
        prepared_data = data_processor.load_and_prepare_data(
            dataset_id=simulation_data.dataset,
            data_dir=DATA_DIR,
            global_settings=simulation_data.global_settings
        )

        # Model değerlendirmesi - SADECE BAŞARI METRİKLERİ
        ml_results = model_factory.run_model_pipeline(
            algorithm_name=simulation_data.algorithm,
            model_params_from_frontend=simulation_data.params,
            data_dict=prepared_data,
            global_settings=simulation_data.global_settings,
            mode="evaluate"  # YENİ: mode parametresi
        )

        unique_run_id = f"{simulation_data.algorithm}_{simulation_data.dataset}_{str(time.time()).replace('.', '')}"

        # EVALUATE response - sadece başarı metrikleri
        response_payload = {
            "configId": simulation_data.params.get("frontend_config_id", unique_run_id),
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success",
            "metrics": ml_results.get("metrics", {}),  # Başarı metrikleri
            "plotData": ml_results.get("plot_data", {}),
            "score_time_seconds": ml_results.get("score_time_seconds"),
            "prediction_performance": ml_results.get("prediction_performance", {}),
            "notes_from_model": ml_results.get("notes", []),
            "overall_status_message": "Model başarıyla değerlendirildi. Tahmin başarısı ölçüldü."
        }

        if not ml_results.get("metrics"):
            response_payload["status"] = "warning"
            response_payload["overall_status_message"] = "Model değerlendirildi ancak metrikler hesaplanamadı."

        print(f"Evaluation Başarı Sonuçları: {response_payload['metrics']}")
        print("--- DEĞERLENDİRME İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except Exception as e:
        # Hata yönetimi aynı...
        return handle_error(e, simulation_data)


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

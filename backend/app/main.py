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

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # sadece frontendden gelen istekleri kabul et
    allow_credentials=True,
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
    /train handles POST requests to the endpoint and starts model 
    training. Parameters from the frontend are kept in simulation_data
"""
@app.post("/train")
async def process_training_request(request: Request, simulation_data: SimulationRequest):

    print("--- Eğitim İsteği Alındı ---")
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
        """
            train the relevant models in the factory with the relevant 
            parameters according to the incoming request
        """
        ml_results = model_factory.run_model_pipeline(
            algorithm_name=simulation_data.algorithm,
            model_params_from_frontend=simulation_data.params,
            data_dict=prepared_data,
            global_settings=simulation_data.global_settings
        )
        # can be useed later for database operations...
        unique_run_id = f"{simulation_data.algorithm}_{simulation_data.dataset}_{str(time.time()).replace('.', '')}"

        # response to be send to the frontend
        response_payload = {
            # frontedn must send this ID:
            "configId": simulation_data.params.get("frontend_config_id", unique_run_id), 
            "modelName": simulation_data.algorithm,
            "datasetId": simulation_data.dataset,
            "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
            "status": "success", # if model training is succesful
            "metrics": ml_results.get("metrics", {}),
            "plotData": ml_results.get("plot_data", {}),
            "fit_time_seconds": ml_results.get("fit_time_seconds"),
            "score_time_seconds": ml_results.get("score_time_seconds"),
            "notes_from_model": ml_results.get("notes", []),
            "overall_status_message": "Model başarıyla eğitildi ve değerlendirildi."
        }
        
        if not ml_results.get("metrics"):
            response_payload["status"] = "warning"
            response_payload["overall_status_message"] = "Model eğitildi ancak metrikler hesaplanamadı veya eksik."
            
            """
                If there are any warning or error messages written in the nootes section 
                while the models are running, add them to the overall status message...
            """
            if ml_results.get("notes"):
                 response_payload["overall_status_message"] += " Detaylar: " + " | ".join(ml_results["notes"])

        print(f"Sonuçlar: {response_payload['metrics']}")
        print("--- Eğitim İsteği Başarıyla Tamamlandı ---")
        return response_payload

    except FileNotFoundError as e:
        print(f"Dosya Bulunamadı Hatası: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        print(f"Değer/Veri Hatası: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e: # for different errors

        import traceback

        print(f"Beklenmedik Genel Hata ('{simulation_data.dataset}', Algoritma: '{simulation_data.algorithm}') işlenirken:")
        traceback.print_exc() # it logs full error stack

        error_detail_for_frontend = f"Sunucuda beklenmedik bir hata oluştu: {type(e).__name__}"
        
        return JSONResponse(
            status_code=500,
            content={
                "configId": simulation_data.params.get("frontend_config_id", f"{simulation_data.algorithm}_{simulation_data.dataset}_error"),
                "modelName": simulation_data.algorithm,
                "datasetId": simulation_data.dataset,
                "datasetName": simulation_data.dataset.replace("_", " ").replace("-", " ").title(),
                "status": "error",
                "metrics": {},
                "plotData": {},
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

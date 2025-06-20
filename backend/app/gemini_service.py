# backend/app/gemini_service.py

import os
import json
import requests
from typing import Dict, Any, Optional

class GeminiAnalysisService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def analyze_chart_data(self, chart_data: Dict[str, Any], chart_type: str, context: Optional[str] = None) -> str:
        """
        Gemini API'ye chart verilerini gönderir ve analiz sonucunu döndürür
        """
        try:
            # Prompt'u hazırla
            prompt = self._create_analysis_prompt(chart_data, chart_type, context)

            # API isteği hazırla
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                }
            }

            headers = {
                "Content-Type": "application/json",
            }

            # API çağrısı
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Gemini'nin response formatını parse et
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text'].strip()

                return "Gemini API'den beklenmeyen response formatı alındı."

            else:
                error_msg = f"Gemini API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                except:
                    pass
                return f"API Hatası: {error_msg}"

        except requests.exceptions.Timeout:
            return "API çağrısı zaman aşımına uğradı. Lütfen tekrar deneyin."
        except requests.exceptions.RequestException as e:
            return f"API isteği başarısız: {str(e)}"
        except Exception as e:
            return f"Beklenmeyen hata: {str(e)}"

    def _create_analysis_prompt(self, chart_data: Dict[str, Any], chart_type: str, context: Optional[str] = None) -> str:
        """
        Chart verilerine göre analiz prompt'u oluşturur
        """

        if chart_type == "radar_chart":
            return f"""
    Analyze this radar chart data for machine learning model comparison:

    DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

    Provide a concise analysis (4-5 sentences max) covering:
    1. What a radar chart shows in ML context
    2. How to interpret the values and shapes
    3. Overall performance comparison of models
    4. Which model shows the most balanced performance

    Write in English, be concise and technical but accessible.
    """

        elif chart_type == "performance_trends":
            return f"""
    Analyze this performance over time data for machine learning models:

    DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

    Provide analysis (5-6 sentences max) covering:
    1. Purpose of tracking performance trends over time
    2. Ideal scenario: high accuracy (top) with low training time (bottom) creates good separation
    3. Identify which model achieves the best accuracy-speed trade-off (maximum separation between metrics)
    4. Determine the best and worst performing models considering both accuracy and training time together
    5. Overall trend interpretation

    Focus on the balance between accuracy and efficiency. Write in English, be concise.
    """

        elif chart_type == "confusion_matrix":
            return f"""
    Analyze this confusion matrix for machine learning model evaluation:

    DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

    Provide analysis (4-5 sentences max) covering:
    1. What a confusion matrix represents in classification
    2. How to interpret the matrix values (diagonal vs off-diagonal)
    3. Model performance assessment based on this specific matrix
    4. Any notable patterns in misclassification

    Write in English, be concise and focus on practical interpretation.
    """

        else:
            # Fallback for other chart types
            return f"""
    Analyze this machine learning performance data:

    DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}
    CHART TYPE: {chart_type}

    Provide a concise analysis (4-5 sentences max) focusing on key insights and model performance patterns. Write in English.
    """

# Global service instance
gemini_service = GeminiAnalysisService()

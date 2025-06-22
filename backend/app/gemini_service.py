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
                    "maxOutputTokens": 2048,  # Increased for more detailed responses
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
        Chart verilerine göre detaylı analiz prompt'u oluşturur
        """

        if chart_type == "bar_chart":
            return f"""
You are an expert machine learning analyst. Analyze this bar chart data for model performance comparison:

DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

Please provide a comprehensive analysis following this structure:

**1. INTRODUCTION & CONTEXT (2-3 sentences)**
- Explain what a bar chart comparison represents in machine learning model evaluation
- Define the purpose of comparing multiple models across different performance metrics

**2. METRICS EXPLANATION (3-4 sentences)**
- Briefly explain what each metric means in classification:
  * Accuracy: Overall correctness rate
  * Precision: How many predicted positives were actually positive
  * Recall: How many actual positives were correctly identified
  * F1-Score: Harmonic mean of precision and recall
  * ROC AUC: Area under the receiver operating characteristic curve

**3. PERFORMANCE ANALYSIS (4-5 sentences)**
- Identify the best performing model overall and why
- Highlight any models that excel in specific metrics
- Point out any notable weaknesses or trade-offs
- Compare consistency across different metrics

**4. PRACTICAL INSIGHTS (2-3 sentences)**
- Provide actionable recommendations based on the results
- Suggest which model to choose based on different use case scenarios

Write in clear, educational English. Be detailed but accessible to both beginners and experts.
"""

        elif chart_type == "radar_chart":
            return f"""
You are an expert machine learning analyst. Analyze this radar chart data for comprehensive model comparison:

DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

Please provide a detailed analysis following this structure:

**1. RADAR CHART FUNDAMENTALS (2-3 sentences)**
- Explain what a radar/spider chart represents in machine learning evaluation
- Describe how to interpret the visual patterns: larger area = better overall performance
- Explain why radar charts are particularly useful for multi-metric model comparison

**2. METRIC DIMENSIONS OVERVIEW (3-4 sentences)**
- Provide context for each metric axis and what optimal values look like
- Explain how the shape of each model's polygon reveals performance characteristics
- Describe what a "balanced" vs "specialized" model looks like in radar format

**3. DETAILED MODEL COMPARISON (5-6 sentences)**
- Identify which model has the largest overall area (best general performance)
- Analyze the shape characteristics: which models are well-rounded vs specialized
- Point out specific strengths and weaknesses for each model
- Identify any models with particularly unbalanced performance profiles
- Compare how consistently each model performs across all metrics

**4. STRATEGIC RECOMMENDATIONS (3-4 sentences)**
- Recommend the most balanced performer for general use cases
- Suggest specialized models for specific requirements (high precision, high recall, etc.)
- Provide guidance on model selection based on business priorities
- Comment on any concerning performance gaps that need attention

Write in an educational, comprehensive style suitable for both technical and non-technical stakeholders.
"""

        elif chart_type == "performance_trends":
            return f"""
You are an expert machine learning analyst. Analyze this performance trends data over time:

DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

Please provide a comprehensive temporal analysis following this structure:

**1. PERFORMANCE TRENDS CONTEXT (2-3 sentences)**
- Explain the importance of tracking ML model performance over time
- Describe what ideal performance trends should look like in production systems
- Define why temporal analysis helps identify model stability and reliability

**2. ACCURACY-EFFICIENCY TRADE-OFF ANALYSIS (4-5 sentences)**
- Explain the fundamental trade-off between model accuracy and training efficiency
- Analyze the relationship between accuracy trends and training time patterns
- Identify models that achieve optimal balance (high accuracy with reasonable training time)
- Point out any models with concerning accuracy degradation or efficiency issues
- Discuss the practical implications of consistent vs volatile performance

**3. TEMPORAL PATTERN INSIGHTS (4-5 sentences)**
- Analyze overall trends: are models improving, stable, or degrading over time?
- Identify any cyclical patterns or anomalies in the performance data
- Compare the consistency of different models across multiple time periods
- Highlight which models show the most stable and reliable performance
- Point out any models with concerning variability that might indicate training instability

**4. MODEL RANKING & RECOMMENDATIONS (3-4 sentences)**
- Rank models based on overall temporal performance considering both accuracy and efficiency
- Recommend the most reliable model for production deployment
- Suggest models that might need hyperparameter tuning or architectural changes
- Provide guidance on monitoring strategies based on observed patterns

**5. OPERATIONAL INSIGHTS (2-3 sentences)**
- Comment on resource allocation implications based on training time patterns
- Suggest optimization strategies for models with poor efficiency trends
- Provide recommendations for continuous monitoring and evaluation protocols

Write in a detailed, analytical style that helps stakeholders understand both technical performance and business implications.
"""

        elif chart_type == "confusion_matrix":
            return f"""
You are an expert machine learning analyst. Analyze this confusion matrix for detailed classification performance evaluation:

DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}

Please provide a comprehensive confusion matrix analysis following this structure:

**1. CONFUSION MATRIX FUNDAMENTALS (3-4 sentences)**
- Explain what a confusion matrix represents in classification problems
- Describe how to read the matrix: rows = actual classes, columns = predicted classes
- Define the significance of diagonal vs off-diagonal elements
- Explain why confusion matrices provide more insight than simple accuracy scores

**2. CLASSIFICATION PERFORMANCE BREAKDOWN (4-5 sentences)**
- Analyze the diagonal elements (correct predictions) for each class
- Identify which classes the model predicts most accurately
- Examine off-diagonal elements to understand misclassification patterns
- Calculate and interpret the overall accuracy from the matrix
- Discuss any class imbalance issues evident in the results

**3. ERROR PATTERN ANALYSIS (4-5 sentences)**
- Identify the most common types of misclassifications
- Analyze whether certain classes are consistently confused with specific others
- Evaluate if misclassification patterns suggest systematic model biases
- Assess the severity of different types of errors (false positives vs false negatives)
- Comment on whether error patterns align with expected class similarities

**4. MODEL QUALITY ASSESSMENT (3-4 sentences)**
- Evaluate overall model performance considering both accuracy and error distribution
- Assess whether the model performs consistently across all classes
- Identify any classes where the model struggles significantly
- Comment on the model's reliability for different classification scenarios

**5. IMPROVEMENT RECOMMENDATIONS (2-3 sentences)**
- Suggest specific strategies to address identified misclassification patterns
- Recommend data collection or model tuning approaches for problematic classes
- Provide guidance on whether this model is ready for production use

Write in an educational, detailed style that helps readers understand both the technical analysis and practical implications for real-world deployment.
"""

        else:
            # Enhanced fallback for other chart types
            return f"""
You are an expert machine learning analyst. Analyze this machine learning performance visualization:

DATA: {json.dumps(chart_data, indent=2, ensure_ascii=False)}
CHART TYPE: {chart_type}
CONTEXT: {context or "General model performance analysis"}

Please provide a comprehensive analysis following this structure:

**1. VISUALIZATION CONTEXT (2-3 sentences)**
- Explain what this type of chart/visualization represents in machine learning
- Describe the purpose and benefits of this analysis approach
- Define what insights this visualization method can reveal

**2. DATA INTERPRETATION (4-5 sentences)**
- Analyze the key patterns visible in the data
- Identify the best and worst performing elements
- Explain any notable relationships or trends in the data
- Discuss the significance of the observed patterns
- Compare relative performance across different dimensions

**3. PERFORMANCE INSIGHTS (3-4 sentences)**
- Provide deeper insights into what the results mean for model selection
- Identify any concerning patterns or exceptional performance
- Discuss trade-offs or balances evident in the data
- Comment on the practical implications of the observed results

**4. ACTIONABLE RECOMMENDATIONS (2-3 sentences)**
- Suggest specific next steps based on the analysis
- Provide guidance on model selection or optimization priorities
- Recommend areas for further investigation or improvement

Write in clear, educational English that provides both technical depth and practical value for decision-making.
"""

# Global service instance
gemini_service = GeminiAnalysisService()

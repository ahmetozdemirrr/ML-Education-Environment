// frontend/src/components/MetricsChart.js - GÜNCELLENMIŞ VERSİYON

import React, { useState, useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Radar, Line, Scatter } from 'react-chartjs-2';
import { MarkdownRenderer } from '../utils/markdownUtils'; // YENİ IMPORT

// Register all Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const getAlgorithmColor = (algorithmName) => {
  const colors = {
    'Decision Tree': '#ef4444',
    'Logistic Regression': '#3b82f6',
    'Random Forest': '#10b981',
    'SVM': '#f59e0b',
    'K-Nearest Neighbor': '#8b5cf6',
    'Artificial Neural Network': '#ec4899',
    'Naive Bayes': '#6b7280',
    'Gradient Boosting': '#06b6d4'
  };

  return colors[algorithmName] || '#64748b';
};

const createUniqueModelNames = (results) => {
  const modelCounts = {};
  return results.map(result => {
    const originalName = result.modelName;
    modelCounts[originalName] = (modelCounts[originalName] || 0) + 1;

    const uniqueName = modelCounts[originalName] === 1
      ? originalName
      : `${originalName} (${modelCounts[originalName]})`;

    return {
      ...result,
      uniqueModelName: uniqueName,
      instanceNumber: modelCounts[originalName]
    };
  });
};

// Etiket kısaltma fonksiyonu
const shortenLabel = (modelName, datasetName) => {
  const modelShort = modelName
    .replace('Logistic Regression', 'LR')
    .replace('Decision Tree', 'DT')
    .replace('Random Forest', 'RF')
    .replace('Support Vector Machine', 'SVM')
    .replace('K-Nearest Neighbor', 'KNN')
    .replace('Artificial Neural Network', 'ANN')
    .replace('Naive Bayes', 'NB')
    .replace('Gradient Boosting', 'GB');

  const datasetShort = datasetName
    .replace('Dataset', '')
    .replace('(Synthetic)', '(Syn)')
    .replace('Two Moons', 'TwoMoons')
    .trim();

  return `${modelShort} (${datasetShort})`;
};

// Gemini API çağrısı
const callGeminiAnalysis = async (chartData, chartType, context = '') => {
  try {
    const response = await fetch('http://localhost:8000/analyze-with-gemini', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chart_data: chartData,
        chart_type: chartType,
        context: context
      })
    });

    const result = await response.json();
    return result.analysis;
  } catch (error) {
    return `Analiz sırasında hata: ${error.message}`;
  }
};

// Metric değeri alma - Backend format'ına uygun
const getMetricValue = (result, metricName) => {
  const metrics = result.metrics || {};

  const possibleKeys = [];

  switch(metricName) {
    case 'accuracy':
      possibleKeys.push('Accuracy', 'accuracy', 'ACCURACY');
      break;
    case 'precision':
      possibleKeys.push('Precision', 'precision', 'PRECISION');
      break;
    case 'recall':
      possibleKeys.push('Recall', 'recall', 'RECALL');
      break;
    case 'f1score':
      possibleKeys.push('F1-Score', 'f1_score', 'f1score', 'F1Score', 'f1-score');
      break;
    case 'roc_auc':
      possibleKeys.push('ROC AUC', 'roc_auc', 'ROC_AUC', 'roc-auc');
      break;
    default:
      possibleKeys.push(
        metricName,
        metricName.toLowerCase(),
        metricName.toUpperCase(),
        metricName.replace('_', '-'),
        metricName.replace('_', ' '),
        metricName.charAt(0).toUpperCase() + metricName.slice(1)
      );
  }

  for (const key of possibleKeys) {
    if (metrics[key] !== undefined && metrics[key] !== null) {
      return typeof metrics[key] === 'number' ? metrics[key] : 0;
    }
  }

  return 0;
};

// Unique results alma
const getUniqueResults = (results) => {
  const seen = new Set();
  return results.filter(result => {
    if (!result.metrics || Object.keys(result.metrics).length === 0) {
      return false;
    }

    const key = `${result.modelName}-${result.datasetName}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

// =================================== //
// METRICS CHART COMPONENT
// =================================== //

const MetricsChart = ({ results, isDarkMode = false  }) => {
  const barChartRef = useRef(null);
  const radarChartRef = useRef(null);
  const [geminiAnalysisBar, setGeminiAnalysisBar] = useState('');
  const [geminiAnalysisRadar, setGeminiAnalysisRadar] = useState('');
  const [isAnalyzingBar, setIsAnalyzingBar] = useState(false);
  const [isAnalyzingRadar, setIsAnalyzingRadar] = useState(false);

  // YENİ: Açıp kapama state'leri
  const [isBarAnalysisExpanded, setIsBarAnalysisExpanded] = useState(true);
  const [isRadarAnalysisExpanded, setIsRadarAnalysisExpanded] = useState(true);

  useEffect(() => {
    return () => {
      if (barChartRef.current) barChartRef.current.destroy();
      if (radarChartRef.current) radarChartRef.current.destroy();
    };
  }, []);

  if (!results || results.length === 0) {
    return <div className="chart-placeholder">No metrics data available</div>;
  }

  const resultsWithMetrics = results.filter(r => r.metrics && Object.keys(r.metrics).length > 0);

  if (resultsWithMetrics.length === 0) {
    return (
      <div className="chart-placeholder">
        <p>No evaluation metrics available for charting.</p>
        <p>Please run some evaluations first to see performance charts.</p>
      </div>
    );
  }

  const uniqueResults = getUniqueResults(resultsWithMetrics);
  const resultsWithUniqueNames = createUniqueModelNames(uniqueResults);

  const metricKeys = ['accuracy', 'precision', 'recall', 'f1score', 'roc_auc'];
  const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'];

  const metricColors = [
    'rgba(239, 68, 68, 0.8)',   // Red
    'rgba(59, 130, 246, 0.8)',  // Blue
    'rgba(245, 158, 11, 0.8)',  // Yellow
    'rgba(75, 85, 99, 0.8)',    // Gray
    'rgba(168, 85, 247, 0.8)'   // Purple
  ];

  const barData = {
    labels: resultsWithUniqueNames.map(r => shortenLabel(r.uniqueModelName, r.datasetName)), // DEĞİŞTİRİLDİ
    datasets: metricKeys.map((metric, index) => {
      const data = resultsWithUniqueNames.map(r => getMetricValue(r, metric));

      return {
        label: metricLabels[index],
        data: data,
        backgroundColor: metricColors[index],
        borderColor: metricColors[index].replace('0.8', '1'),
        borderWidth: 1
      };
    })
  };

  // Radar Chart Data
  const generateColors = (count) => {
    const colors = [];
    for (let i = 0; i < count; i++) {
      const hue = (i * 360 / count) % 360;
      colors.push({
        border: `hsl(${hue}, 70%, 50%)`,
        bg: `hsla(${hue}, 70%, 50%, 0.2)`
      });
    }
    return colors;
  };

  const colors = generateColors(resultsWithUniqueNames.length);

  const radarData = {
    labels: metricLabels,
    datasets: resultsWithUniqueNames.map((result, index) => {
      const color = colors[index];
      const data = metricKeys.map(metric => getMetricValue(result, metric));

      return {
        label: shortenLabel(result.uniqueModelName, result.datasetName),
        data: data,
        borderColor: color.border,
        backgroundColor: color.bg,
        borderWidth: 2,
        pointBackgroundColor: color.border,
        pointBorderColor: '#fff',
        pointBorderWidth: 1,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: color.border,
        pointHoverBorderWidth: 2
      };
    })
  };

  return (
    <div className="metrics-chart-container">
      <h4>Performance Metrics Comparison</h4>

      {/* Bar Chart Section */}
      <div className="chart-section">
        <h5>Bar Chart Comparison</h5>

        {/* Chart Container - SADECE CHART */}
        <div style={{ height: '350px', marginBottom: '20px', backgroundColor: isDarkMode ? '#1f2937' : '#ffffff'}}>
          <Bar
            ref={barChartRef}
            data={barData}
            options={getChartOptions(isDarkMode, {
              interaction: { intersect: false },
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Model Performance Metrics' }
              },
              scales: {
                y: { beginAtZero: true, max: 1.1 },
                x: {
                  ticks: {
                    maxRotation: 45,
                    minRotation: 45,
                    font: { size: 10 }
                  }
                }
              }
            })}
          />
        </div>

        {/* Değerler Tablosu */}
        <div className="chart-values-section">
          <h6 style={{ margin: '0 0 12px 0', color: '#374151', fontSize: '1rem' }}>Exact Values</h6>
          <div className="values-table-wrapper">
            <table className="chart-values-table">
              <thead>
                <tr>
                  <th style={{ minWidth: '200px' }}>Model (Dataset)</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>ROC AUC</th>
                </tr>
              </thead>
              <tbody>
                {resultsWithUniqueNames.map((result, index) => (
                  <tr key={index}>
                    <td className="model-name-cell">
                      <div style={{ fontSize: '0.875rem', fontWeight: '600' }}>
                        {result.uniqueModelName}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                        Instance #{result.instanceNumber || 1} | {result.datasetName}
                      </div>
                    </td>
                    {metricKeys.map((metric, metricIndex) => {
                      const value = getMetricValue(result, metric);
                      return (
                        <td key={metricIndex} className="metric-value-cell">
                          {value.toFixed(3)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Bar Chart Gemini Analysis - GÜNCELLENMIŞ */}
        <div className="gemini-analysis-section">
          <button
            className="gemini-explain-btn"
            onClick={async () => {
              setIsAnalyzingBar(true);
              const chartData = {
              models: resultsWithUniqueNames.map(r => ({
                  name: shortenLabel(r.uniqueModelName, r.datasetName),
                  metrics: {
                    accuracy: getMetricValue(r, 'accuracy'),
                    precision: getMetricValue(r, 'precision'),
                    recall: getMetricValue(r, 'recall'),
                    f1_score: getMetricValue(r, 'f1score'),
                    roc_auc: getMetricValue(r, 'roc_auc')
                  }
                }))
              };
              const analysis = await callGeminiAnalysis(chartData, 'bar_chart', 'Bar chart - model performans karşılaştırması');
              setGeminiAnalysisBar(analysis);
              setIsAnalyzingBar(false);
              setIsBarAnalysisExpanded(true); // Analiz geldiğinde otomatik aç
            }}
            disabled={isAnalyzingBar}
          >
            {isAnalyzingBar ? '🤖 Analiz ediliyor...' : '🤖 Explain With Gemini'}
          </button>

          {geminiAnalysisBar && (
            <div className="gemini-analysis-header">
              <h6 style={{ margin: 0, color: '#fbbf24', fontWeight: 600, fontSize: '15px' }}>🤖 Gemini AI Analysis:</h6>
              <button
                className="gemini-toggle-btn"
                onClick={() => setIsBarAnalysisExpanded(!isBarAnalysisExpanded)}
              >
                <span className="gemini-toggle-text">
                  {isBarAnalysisExpanded ? 'Hide' : 'Show'}
                </span>
                <span className="gemini-toggle-icon">
                  {isBarAnalysisExpanded ? '⊟' : '⊞'}
                </span>
              </button>
            </div>
          )}

          {geminiAnalysisBar && (
            <div className={`gemini-analysis-result ${isBarAnalysisExpanded ? 'expanded' : 'collapsed'}`}>
              <MarkdownRenderer content={geminiAnalysisBar} />
            </div>
          )}
        </div>
      </div>

      {/* Radar Chart Section */}
      <div className="chart-section">
        <h5>Radar Chart Comparison</h5>

        {/* Chart Container - SADECE CHART */}
        <div style={{ height: '400px' }}>
          <Radar
            ref={radarChartRef}
            data={radarData}
            options={getChartOptions(isDarkMode, {
              plugins: {
                legend: {
                  position: 'bottom',
                  labels: {
                    boxWidth: 12,
                    font: { size: 10 },
                    padding: 8,
                    usePointStyle: true
                  }
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      return `${context.dataset.label}: ${context.parsed.r.toFixed(3)}`;
                    }
                  }
                }
              },
              scales: {
                r: {
                  beginAtZero: true,
                  max: 1,
                  min: 0,
                  ticks: {
                    stepSize: 0.2,
                    font: { size: 10 },
                    backdropColor: 'transparent'
                  }
                }
              },
              elements: {
                line: { borderWidth: 2 },
                point: { radius: 3, hoverRadius: 5 }
              },
              interaction: {
                intersect: false,
                mode: 'nearest'
              }
            })}
          />
        </div>

        {/* Radar Chart Gemini Analysis - GÜNCELLENMIŞ */}
        <div className="gemini-analysis-section">
          <button
            className="gemini-explain-btn"
            onClick={async () => {
              setIsAnalyzingRadar(true); // DÜZELTİLDİ
              const chartData = {
                models: resultsWithUniqueNames.map(r => ({
                  name: shortenLabel(r.uniqueModelName, r.datasetName),
                  metrics: {
                    accuracy: getMetricValue(r, 'accuracy'),
                    precision: getMetricValue(r, 'precision'),
                    recall: getMetricValue(r, 'recall'),
                    f1_score: getMetricValue(r, 'f1score'),
                    roc_auc: getMetricValue(r, 'roc_auc')
                  }
                }))
              };
              const analysis = await callGeminiAnalysis(chartData, 'radar_chart', 'Radar chart - tüm modellerin performans karşılaştırması');
              setGeminiAnalysisRadar(analysis);
              setIsAnalyzingRadar(false);
              setIsRadarAnalysisExpanded(true); // Analiz geldiğinde otomatik aç
            }}
            disabled={isAnalyzingRadar}
          >
            {isAnalyzingRadar ? '🤖 Analiz ediliyor...' : '🤖 Explain With Gemini'}
          </button>

          {geminiAnalysisRadar && (
            <div className="gemini-analysis-header">
              <h6 style={{ margin: 0, color: '#fbbf24', fontWeight: 600, fontSize: '15px' }}>🤖 Gemini AI Analysis:</h6>
              <button
                className="gemini-toggle-btn"
                onClick={() => setIsRadarAnalysisExpanded(!isRadarAnalysisExpanded)}
              >
                <span className="gemini-toggle-text">
                  {isRadarAnalysisExpanded ? 'Hide' : 'Show'}
                </span>
                <span className="gemini-toggle-icon">
                  {isRadarAnalysisExpanded ? '⊟' : '⊞'}
                </span>
              </button>
            </div>
          )}

          {geminiAnalysisRadar && (
            <div className={`gemini-analysis-result ${isRadarAnalysisExpanded ? 'expanded' : 'collapsed'}`}>
              <MarkdownRenderer content={geminiAnalysisRadar} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// =================================== //
// PERFORMANCE CHART COMPONENT
// =================================== //

const PerformanceChart = ({ results, isDarkMode = false }) => {
  const scatterChartRef = useRef(null);
  const lineChartRef = useRef(null);
  const [geminiAnalysis, setGeminiAnalysis] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // YENİ: Açıp kapama state'i
  const [isPerformanceAnalysisExpanded, setIsPerformanceAnalysisExpanded] = useState(true);

  useEffect(() => {
    return () => {
      if (scatterChartRef.current) scatterChartRef.current.destroy();
      if (lineChartRef.current) lineChartRef.current.destroy();
    };
  }, []);

  if (!results || results.length === 0) {
    return <div className="chart-placeholder">No performance data available</div>;
  }

  // Accuracy değerini alma
  const getAccuracy = (result) => {
    const metrics = result.metrics || {};
    return metrics.Accuracy || metrics.accuracy || metrics.ACCURACY || 0;
  };

  // Dataset bazında gruplandırma
  const datasetGroups = {};
  results.forEach(result => {
    const datasetName = result.datasetName;
    if (!datasetGroups[datasetName]) {
      datasetGroups[datasetName] = [];
    }
    datasetGroups[datasetName].push(result);
  });

  // Model renkleri
  const modelColors = {
    'Decision Tree': 'rgba(255, 99, 132, 0.8)',
    'Logistic Regression': 'rgba(54, 162, 235, 0.8)',
    'SVM': 'rgba(255, 205, 86, 0.8)',
    'K-Nearest Neighbor': 'rgba(75, 192, 192, 0.8)',
    'Artificial Neural Network': 'rgba(153, 102, 255, 0.8)',
    'Random Forest': 'rgba(255, 159, 64, 0.8)',
    'Naive Bayes': 'rgba(199, 199, 199, 0.8)',
    'Gradient Boosting': 'rgba(83, 102, 255, 0.8)'
  };

  const modelBorderColors = {
    'Decision Tree': 'rgba(255, 99, 132, 1)',
    'Logistic Regression': 'rgba(54, 162, 235, 1)',
    'SVM': 'rgba(255, 205, 86, 1)',
    'K-Nearest Neighbor': 'rgba(75, 192, 192, 1)',
    'Artificial Neural Network': 'rgba(153, 102, 255, 1)',
    'Random Forest': 'rgba(255, 159, 64, 1)',
    'Naive Bayes': 'rgba(199, 199, 199, 1)',
    'Gradient Boosting': 'rgba(83, 102, 255, 1)'
  };

  // Dataset scatter chart render
  const renderScatterChart = (datasetName, datasetResults) => {
    const modelGroups = {};
    datasetResults.forEach(result => {
      const modelName = result.modelName;
      if (!modelGroups[modelName]) {
        modelGroups[modelName] = [];
      }
      modelGroups[modelName].push({
        x: result.fit_time_seconds || result.score_time_seconds || 0,
        y: getAccuracy(result),
        label: modelName,
        dataset: datasetName,
        memory: result.memory_usage_mb || 0
      });
    });

    const datasets = Object.keys(modelGroups).map(modelName => ({
      label: modelName,
      data: modelGroups[modelName],
      backgroundColor: modelColors[modelName] || 'rgba(128, 128, 128, 0.8)',
      borderColor: modelBorderColors[modelName] || 'rgba(128, 128, 128, 1)',
      borderWidth: 1,
      pointRadius: 4,
      pointHoverRadius: 5
    }));

    return (
      <div key={`scatter-${datasetName}`} className="chart-section dataset-chart">
        <h5>Accuracy vs Training Time - {datasetName}</h5>

        {/* Chart Container - SADECE CHART */}
        <div style={{ height: '350px' }}>
          <Scatter
            data={{ datasets }}
            options={getChartOptions(isDarkMode, {
              interaction: { intersect: false },
              plugins: {
                legend: {
                  position: 'top',
                  display: true,
                  labels: { usePointStyle: true, padding: 20 }
                },
                title: { display: true, text: `Performance Trade-offs for ${datasetName}` },
                tooltip: {
                  callbacks: {
                    label: (context) => {
                      const point = context.raw;
                      return `${point.label}: Accuracy: ${point.y.toFixed(4)}, Time: ${point.x.toFixed(4)}s`;
                    }
                  }
                }
              },
              scales: {
                x: { title: { display: true, text: 'Training Time (seconds)' }, beginAtZero: true },
                y: { title: { display: true, text: 'Accuracy' }, beginAtZero: true, max: 1 }
              }
            })}
          />
        </div>
      </div>
    );
  };
  const resultsWithUniqueNames = createUniqueModelNames(results);

  // Line chart data
  const timeSeriesData = {
    labels: resultsWithUniqueNames.map(r => shortenLabel(r.uniqueModelName, r.datasetName)),

    datasets: [
      {
        label: 'Accuracy',
        data: resultsWithUniqueNames.map(r => getAccuracy(r)),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        yAxisID: 'y'
      },
      {
        label: 'Training Time (s)',
        data: resultsWithUniqueNames.map(r => r.fit_time_seconds || r.score_time_seconds || 0), // DEĞİŞTİRİLDİ
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        yAxisID: 'y1'
      }
    ]
  };

  return (
    <div className="performance-chart-container">
      <h4>Performance Analysis</h4>

      {/* Dataset bazında scatter plotlar */}
      <div className="dataset-charts-grid">
        {Object.keys(datasetGroups).map(datasetName =>
          renderScatterChart(datasetName, datasetGroups[datasetName])
        )}
      </div>

      {/* Genel trend analizi */}
      <div className="chart-section">
        <h5>Performance Over Time (All Results)</h5>

        {/* Chart Container - SADECE CHART */}
        <div style={{ height: '300px' }}>
          <Line
            ref={lineChartRef}
            data={timeSeriesData}
            options={getChartOptions(isDarkMode, {
              interaction: { intersect: false },
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Accuracy and Training Time Trends' }
              },
              scales: {
                y: {
                  type: 'linear',
                  display: true,
                  position: 'left',
                  title: { display: true, text: 'Accuracy' },
                  beginAtZero: true,
                  max: 1
                },
                y1: {
                  type: 'linear',
                  display: true,
                  position: 'right',
                  title: { display: true, text: 'Training Time (s)' },
                  grid: { drawOnChartArea: false },
                  beginAtZero: true
                }
              }
            })}
          />
        </div>

        {/* Performance Trends Gemini Analysis - GÜNCELLENMIŞ */}
        <div className="gemini-analysis-section">
          <button
            className="gemini-explain-btn"
            onClick={async () => {
              setIsAnalyzing(true);
              const trendsData = {
                time_series: results.map((result, index) => ({
                  run: index + 1,
                  model: result.modelName,
                  dataset: result.datasetName,
                  accuracy: getAccuracy(result),
                  time: result.fit_time_seconds || result.score_time_seconds || 0
                }))
              };
              const analysis = await callGeminiAnalysis(trendsData, 'performance_trends', 'Zaman içinde performans trendleri analizi');
              setGeminiAnalysis(analysis);
              setIsAnalyzing(false);
              setIsPerformanceAnalysisExpanded(true); // Analiz geldiğinde otomatik aç
            }}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '🤖 Analiz ediliyor...' : '🤖 Explain With Gemini'}
          </button>

          {geminiAnalysis && (
            <div className="gemini-analysis-header">
              <h6 style={{ margin: 0, color: '#fbbf24', fontWeight: 600, fontSize: '15px' }}>🤖 Gemini AI Analysis:</h6>
              <button
                className="gemini-toggle-btn"
                onClick={() => setIsPerformanceAnalysisExpanded(!isPerformanceAnalysisExpanded)}
              >
                <span className="gemini-toggle-text">
                  {isPerformanceAnalysisExpanded ? 'Hide' : 'Show'}
                </span>
                <span className="gemini-toggle-icon">
                  {isPerformanceAnalysisExpanded ? '⊟' : '⊞'}
                </span>
              </button>
            </div>
          )}

          {geminiAnalysis && (
            <div className={`gemini-analysis-result ${isPerformanceAnalysisExpanded ? 'expanded' : 'collapsed'}`}>
              <MarkdownRenderer content={geminiAnalysis} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// =================================== //
// CONFUSION MATRIX COMPONENT
// =================================== //

const ConfusionMatrix = ({ results, isDarkMode = false  }) => {
  const [geminiAnalysis, setGeminiAnalysis] = useState({});
  const [isAnalyzing, setIsAnalyzing] = useState({});

  // YENİ: Her matrix için açıp kapama state'i
  const [expandedAnalysis, setExpandedAnalysis] = useState({});

  if (!results || results.length === 0) {
    return <div className="chart-placeholder">No confusion matrix data available</div>;
  }

  const resultsWithMatrix = results.filter(r => {
    const locations = [
      r.plot_data?.confusion_matrix,
      r.enhanced_results?.visualization_data?.confusion_matrix,
      r.enhanced_results?.plot_data?.confusion_matrix,
      r.plotData?.confusion_matrix
    ];

    return locations.some(location =>
      location &&
      location.matrix &&
      location.labels &&
      Array.isArray(location.matrix) &&
      Array.isArray(location.labels)
    );
  });

  if (resultsWithMatrix.length === 0) {
    return (
      <div className="confusion-matrix-container">
        <h4>Confusion Matrix</h4>
        <div className="no-matrix-data">
          <p>Confusion matrix data not available</p>
          <p>Backend might not be generating plot data correctly</p>
        </div>
      </div>
    );
  }

  const renderSingleConfusionMatrix = (result) => {
    let matrixData = null;
    const locations = [
      result.plot_data?.confusion_matrix,
      result.enhanced_results?.visualization_data?.confusion_matrix,
      result.enhanced_results?.plot_data?.confusion_matrix,
      result.plotData?.confusion_matrix
    ];

    for (const location of locations) {
      if (location && location.matrix && location.labels) {
        matrixData = location;
        break;
      }
    }

    if (!matrixData) return null;

    const matrix = matrixData.matrix;
    const labels = matrixData.labels;
    const resultId = result.configId || result.modelName;

    const getIntensityColor = (value, max) => {
      const intensity = value / max;
      return `rgba(59, 130, 246, ${0.1 + intensity * 0.8})`;
    };

    const maxValue = Math.max(...matrix.flat());

    return (
      <div key={resultId} className="single-confusion-matrix">
        {/* Matrix Info */}
        <div className="matrix-info">
          <h5>{result.modelName} on {result.datasetName}</h5>
        </div>

        {/* Matrix Wrapper */}
        <div className="matrix-wrapper">
          <div className="matrix-labels y-labels">
            <div className="label-header">Actual</div>
            {labels.map((label, idx) => (
              <div key={idx} className="label">{label}</div>
            ))}
          </div>

          <div className="matrix-content">
            <div className="matrix-labels x-labels">
              <div className="label-header">Predicted</div>
              {labels.map((label, idx) => (
                <div key={idx} className="label">{label}</div>
              ))}
            </div>

            <div className="matrix-grid">
              {matrix.map((row, rowIdx) => (
                <div key={rowIdx} className="matrix-row">
                  {row.map((value, colIdx) => (
                    <div
                      key={`${rowIdx}-${colIdx}`}
                      className={`matrix-cell ${rowIdx === colIdx ? 'diagonal' : ''}`}
                      style={{ backgroundColor: getIntensityColor(value, maxValue) }}
                    >
                      {value}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Confusion Matrix Gemini Analysis - GÜNCELLENMIŞ */}
        <div className="gemini-analysis-section">
          <button
            className="gemini-explain-btn"
            onClick={async () => {
              setIsAnalyzing(prev => ({ ...prev, [resultId]: true }));
              const confusionData = {
                matrix: matrix,
                labels: labels,
                model: result.modelName,
                dataset: result.datasetName,
                total_predictions: matrix.flat().reduce((a, b) => a + b, 0),
                correct_predictions: matrix.reduce((sum, row, idx) => sum + row[idx], 0),
                accuracy: ((matrix.reduce((sum, row, idx) => sum + row[idx], 0) / matrix.flat().reduce((a, b) => a + b, 0)) * 100).toFixed(1)
              };
              const analysis = await callGeminiAnalysis(confusionData, 'confusion_matrix', `${result.modelName} modeli için confusion matrix analizi`);
              setGeminiAnalysis(prev => ({ ...prev, [resultId]: analysis }));
              setIsAnalyzing(prev => ({ ...prev, [resultId]: false }));
              setExpandedAnalysis(prev => ({ ...prev, [resultId]: true })); // Analiz geldiğinde otomatik aç
            }}
            disabled={isAnalyzing[resultId]}
          >
            {isAnalyzing[resultId] ? '🤖 Analiz ediliyor...' : '🤖 Explain With Gemini'}
          </button>

          {geminiAnalysis[resultId] && (
            <div className="gemini-analysis-header">
              <h6 style={{ margin: 0, color: '#fbbf24', fontWeight: 600, fontSize: '15px' }}>🤖 Gemini AI Analysis:</h6>
              <button
                className="gemini-toggle-btn"
                onClick={() => setExpandedAnalysis(prev => ({
                  ...prev,
                  [resultId]: !prev[resultId]
                }))}
              >
                <span className="gemini-toggle-text">
                  {expandedAnalysis[resultId] ? 'Hide' : 'Show'}
                </span>
                <span className="gemini-toggle-icon">
                  {expandedAnalysis[resultId] ? '⊟' : '⊞'}
                </span>
              </button>
            </div>
          )}

          {geminiAnalysis[resultId] && (
            <div className={`gemini-analysis-result ${expandedAnalysis[resultId] ? 'expanded' : 'collapsed'}`}>
              <MarkdownRenderer content={geminiAnalysis[resultId]} />
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="confusion-matrix-container">
      <h4>Confusion Matrix</h4>
      <div className="confusion-matrices-grid">
        {resultsWithMatrix.map(result => renderSingleConfusionMatrix(result))}
      </div>
    </div>
  );
};

// =================================== //
// COMPARISON TABLE COMPONENT (DEĞİŞİKLİK YOK)
// =================================== //

const ComparisonTable = ({ results }) => {
  const [sortColumn, setSortColumn] = useState('accuracy');
  const [sortDirection, setSortDirection] = useState('desc');

  if (!results || results.length === 0) {
    return (
      <div className="comparison-table-container">
        <div className="no-comparison-data">
          <p>No results selected for comparison</p>
          <p>Select results from the Details tab to compare them here</p>
        </div>
      </div>
    );
  }

  // Metrik değerlerini almak için fonksiyonlar
  const getMetricValue = (result, metricType) => {
    const metrics = result.metrics || {};

    switch(metricType) {
      case 'accuracy':
        return metrics.Accuracy || metrics.accuracy || metrics.ACCURACY || 0;
      case 'precision':
        return metrics.Precision || metrics.precision || metrics.PRECISION || 0;
      case 'recall':
        return metrics.Recall || metrics.recall || metrics.RECALL || 0;
      case 'f1_score':
        return metrics['F1-Score'] || metrics.f1_score || metrics.f1score || metrics['F1Score'] || 0;
      case 'roc_auc':
        return metrics['ROC AUC'] || metrics.roc_auc || metrics['ROC_AUC'] || 0;
      default:
        return 0;
    }
  };

  // Dataset bazında gruplama
  const datasetGroups = {};
  results.forEach(result => {
    const datasetName = result.datasetName;
    if (!datasetGroups[datasetName]) {
      datasetGroups[datasetName] = [];
    }
    datasetGroups[datasetName].push(result);
  });

  // Her dataset için en iyi modeli bulma
  const getBestModelForDataset = (datasetResults) => {
    return datasetResults.reduce((best, current) => {
      const currentAcc = getMetricValue(current, 'accuracy');
      const bestAcc = getMetricValue(best, 'accuracy');
      return currentAcc > bestAcc ? current : best;
    });
  };

  // Sorting fonksiyonu
  const getSortedResults = (datasetResults) => {
    return [...datasetResults].sort((a, b) => {
      let aVal, bVal;

      switch (sortColumn) {
        case 'accuracy':
          aVal = getMetricValue(a, 'accuracy');
          bVal = getMetricValue(b, 'accuracy');
          break;
        case 'precision':
          aVal = getMetricValue(a, 'precision');
          bVal = getMetricValue(b, 'precision');
          break;
        case 'recall':
          aVal = getMetricValue(a, 'recall');
          bVal = getMetricValue(b, 'recall');
          break;
        case 'f1_score':
          aVal = getMetricValue(a, 'f1_score');
          bVal = getMetricValue(b, 'f1_score');
          break;
        case 'time':
          aVal = a.fit_time_seconds || a.score_time_seconds || 0;
          bVal = b.fit_time_seconds || b.score_time_seconds || 0;
          break;
        case 'model':
          aVal = a.modelName || '';
          bVal = b.modelName || '';
          break;
        default:
          aVal = 0;
          bVal = 0;
      }

      if (typeof aVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      } else {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
    });
  };

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ column }) => {
    if (sortColumn !== column) return <span className="sort-icon">⇅</span>;
    return sortDirection === 'asc' ? <span className="sort-icon">↑</span> : <span className="sort-icon">↓</span>;
  };

  // Dataset bazında tablo render etme
  const renderDatasetTable = (datasetName, datasetResults) => {
    const sortedResults = getSortedResults(datasetResults);
    const bestModel = getBestModelForDataset(datasetResults);

    // Dataset istatistikleri
    const avgAccuracy = datasetResults.reduce((sum, r) => sum + getMetricValue(r, 'accuracy'), 0) / datasetResults.length;
    const bestAccuracy = Math.max(...datasetResults.map(r => getMetricValue(r, 'accuracy')));

    return (
      <div key={datasetName} className="dataset-comparison-section">
        {/* Dataset Header */}
        <div className="dataset-header">
          <div className="dataset-info">
            <h4>📊 {datasetName}</h4>
            <div className="dataset-stats">
              <span className="stat-chip">
                <span className="stat-label">Models:</span>
                <span className="stat-value">{datasetResults.length}</span>
              </span>
              <span className="stat-chip">
                <span className="stat-label">Avg Accuracy:</span>
                <span className="stat-value">{avgAccuracy.toFixed(3)}</span>
              </span>
              <span className="stat-chip">
                <span className="stat-label">Best Accuracy:</span>
                <span className="stat-value">{bestAccuracy.toFixed(3)}</span>
              </span>
            </div>
          </div>

          {/* Best Performer for this dataset */}
          <div className="dataset-best-performer">
            <span className="best-crown">👑</span>
            <div className="best-info">
              <span className="best-label">Best Performer:</span>
              <span className="best-model">{bestModel.modelName}</span>
              <span className="best-accuracy">
                ({getMetricValue(bestModel, 'accuracy').toFixed(3)} accuracy)
              </span>
            </div>
          </div>
        </div>

        {/* Dataset Table */}
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('model')} className="sortable">
                  Model <SortIcon column="model" />
                </th>
                <th onClick={() => handleSort('accuracy')} className="sortable">
                  Accuracy <SortIcon column="accuracy" />
                </th>
                <th onClick={() => handleSort('precision')} className="sortable">
                  Precision <SortIcon column="precision" />
                </th>
                <th onClick={() => handleSort('recall')} className="sortable">
                  Recall <SortIcon column="recall" />
                </th>
                <th onClick={() => handleSort('f1_score')} className="sortable">
                  F1-Score <SortIcon column="f1_score" />
                </th>
                <th onClick={() => handleSort('time')} className="sortable">
                  Time (s) <SortIcon column="time" />
                </th>
                <th>Memory (MB)</th>
                <th>Cache</th>
                <th>Performance</th>
                <th>Rank</th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map((result, index) => {
                const accuracy = getMetricValue(result, 'accuracy');
                const precision = getMetricValue(result, 'precision');
                const recall = getMetricValue(result, 'recall');
                const f1Score = getMetricValue(result, 'f1_score');
                const time = result.fit_time_seconds || result.score_time_seconds || 0;
                const memory = result.memory_usage_mb || 0;

                const performanceLevel = accuracy >= 0.95 ? 'excellent' :
                                       accuracy >= 0.90 ? 'good' :
                                       accuracy >= 0.80 ? 'fair' : 'poor';

                const isBest = result.configId === bestModel.configId;

                return (
                  <tr
                    key={result.configId || index}
                    className={isBest ? 'best-row dataset-best' : ''}
                  >
                    <td className="model-cell">
                      <div className="model-name">{result.modelName}</div>
                      {isBest && (
                        <div className="dataset-best-badge">
                          <span className="crown-icon">👑</span>
                          BEST FOR {datasetName.toUpperCase()}
                        </div>
                      )}
                    </td>
                    <td className="metric-cell">
                      <div className="metric-value">{accuracy.toFixed(4)}</div>
                      <div className="metric-bar">
                        <div
                          className="metric-fill accuracy"
                          style={{ width: `${accuracy * 100}%` }}
                        ></div>
                      </div>
                    </td>
                    <td className="metric-cell">
                      <div className="metric-value">{precision.toFixed(4)}</div>
                      <div className="metric-bar">
                        <div
                          className="metric-fill precision"
                          style={{ width: `${precision * 100}%` }}
                        ></div>
                      </div>
                    </td>
                    <td className="metric-cell">
                      <div className="metric-value">{recall.toFixed(4)}</div>
                      <div className="metric-bar">
                        <div
                          className="metric-fill recall"
                          style={{ width: `${recall * 100}%` }}
                        ></div>
                      </div>
                    </td>
                    <td className="metric-cell">
                      <div className="metric-value">{f1Score.toFixed(4)}</div>
                      <div className="metric-bar">
                        <div
                          className="metric-fill f1score"
                          style={{ width: `${f1Score * 100}%` }}
                        ></div>
                      </div>
                    </td>
                    <td>{time.toFixed(3)}</td>
                    <td>{memory.toFixed(1)}</td>
                    <td>
                      <span className={`cache-badge ${result.from_cache ? 'hit' : 'miss'}`}>
                        {result.from_cache ? 'HIT' : 'MISS'}
                      </span>
                    </td>
                    <td>
                      <span className={`performance-badge ${performanceLevel}`}>
                        {performanceLevel.toUpperCase()}
                      </span>
                    </td>
                    <td>
                      <span className={`rank-badge rank-${index + 1}`}>
                        #{index + 1}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Genel özet istatistikler
  const totalResults = results.length;
  const datasetsCount = Object.keys(datasetGroups).length;
  const overallBest = results.reduce((best, current) => {
    const currentAcc = getMetricValue(current, 'accuracy');
    const bestAcc = getMetricValue(best, 'accuracy');
    return currentAcc > bestAcc ? current : best;
  });

  return (
    <div className="comparison-tab">
      {/* Global Header */}
      <div className="comparison-global-header">
        <div className="global-info">
          <h3>🏆 Model Performance Comparison</h3>
          <div className="global-stats">
            <span className="global-stat">
              <span className="stat-label">Total Models:</span>
              <span className="stat-value">{totalResults}</span>
            </span>
            <span className="global-stat">
              <span className="stat-label">Datasets:</span>
              <span className="stat-value">{datasetsCount}</span>
            </span>
            <span className="global-stat">
              <span className="stat-label">Overall Champion:</span>
              <span className="stat-value champion">{overallBest.modelName}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Dataset Sections */}
      <div className="datasets-comparison-container">
        {Object.keys(datasetGroups)
          .sort()
          .map(datasetName => renderDatasetTable(datasetName, datasetGroups[datasetName]))
        }
      </div>
    </div>
  );
};

// Dark mode için Chart.js tema ayarları
const getChartTheme = (isDarkMode) => {
  if (isDarkMode) {
    return {
      backgroundColor: '#1f2937',
      textColor: '#e5e7eb',
      gridColor: '#4b5563',
      borderColor: '#6b7280',
      tickColor: '#d1d5db',
      tooltipBackground: '#374151',
      tooltipBorder: '#4b5563',
      tooltipText: '#f3f4f6'
    };
  } else {
    return {
      backgroundColor: '#ffffff',
      textColor: '#374151',
      gridColor: '#e5e7eb',
      borderColor: '#d1d5db',
      tickColor: '#6b7280',
      tooltipBackground: '#ffffff',
      tooltipBorder: '#e5e7eb',
      tooltipText: '#1f2937'
    };
  }
};

// Chart.js için dark mode options
const getChartOptions = (isDarkMode, baseOptions = {}) => {
  const theme = getChartTheme(isDarkMode);

  return {
    ...baseOptions,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      ...baseOptions.plugins,
      legend: {
        ...baseOptions.plugins?.legend,
        labels: {
          ...baseOptions.plugins?.legend?.labels,
          color: theme.textColor,
          font: {
            size: 12,
            family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
          }
        }
      },
      tooltip: {
        ...baseOptions.plugins?.tooltip,
        backgroundColor: theme.tooltipBackground,
        titleColor: theme.tooltipText,
        bodyColor: theme.tooltipText,
        borderColor: theme.tooltipBorder,
        borderWidth: 1,
        cornerRadius: 6
      }
    },
    scales: {
      ...baseOptions.scales,
      x: {
        ...baseOptions.scales?.x,
        ticks: {
          ...baseOptions.scales?.x?.ticks,
          color: theme.tickColor,
          font: { size: 11 }
        },
        grid: {
          ...baseOptions.scales?.x?.grid,
          color: theme.gridColor,
          borderColor: theme.borderColor
        },
        title: {
          ...baseOptions.scales?.x?.title,
          color: theme.textColor,
          font: { size: 12, weight: 'bold' }
        }
      },
      y: {
        ...baseOptions.scales?.y?.ticks,
        ticks: {
          color: theme.tickColor,
          font: { size: 11 }
        },
        grid: {
          ...baseOptions.scales?.y?.grid,
          color: theme.gridColor,
          borderColor: theme.borderColor
        },
        title: {
          ...baseOptions.scales?.y?.title,
          color: theme.textColor,
          font: { size: 12, weight: 'bold' }
        }
      },
      r: {
        ...baseOptions.scales?.r,
        ticks: {
          ...baseOptions.scales?.r?.ticks,
          color: theme.tickColor,
          font: { size: 10 },
          backdropColor: 'transparent'
        },
        grid: {
          ...baseOptions.scales?.r?.grid,
          color: theme.gridColor
        },
        angleLines: {
          ...baseOptions.scales?.r?.angleLines,
          color: theme.gridColor
        },
        pointLabels: {
          ...baseOptions.scales?.r?.pointLabels,
          color: theme.textColor,
          font: { size: 11, weight: '500' }
        }
      }
    }
  };
};

// =================================== //
// EXPORTS
// =================================== //

export default MetricsChart;
export { PerformanceChart, ConfusionMatrix, ComparisonTable };

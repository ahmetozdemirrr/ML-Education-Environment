// frontend/src/components/ResultsViewer.js - MARKDOWN SUPPORT ADDED + CODE VIEWER INTEGRATION

import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';
import { Bar, Pie, Line } from 'react-chartjs-2';
import MetricsChart from './MetricsChart';
import { PerformanceChart, ConfusionMatrix, ComparisonTable, ROCCurveChart } from './MetricsChart';
import { MarkdownRenderer } from '../utils/markdownUtils'; // YENİ IMPORT
import CodeViewer from './CodeViewer'; // YENİ IMPORT - CODE VIEWER
import TrainingAnimation from './TrainingAnimation'; // YENİ
import AlgorithmRace from './AlgorithmRace'; // YENİ

import DatasetExplorationAnimation from './animations/DatasetExplorationAnimation';

import './ResultsViewer.css';

// Register ALL Chart.js components properly
ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler
);

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

const ResultsViewer = ({ results = [], isLoading = false, onClearResults, isDarkMode = false }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedResults, setSelectedResults] = useState([]);
  const [filterBy, setFilterBy] = useState('all');
  const [sortBy, setSortBy] = useState('timestamp');
  const [expandedResults, setExpandedResults] = useState(new Set());

  // YENİ: Code Viewer State - SOURCE CODE MODAL İÇİN
  const [codeViewerOpen, setCodeViewerOpen] = useState(false);
  const [selectedModelForCode, setSelectedModelForCode] = useState(null);

  // Move all refs to component level - FIXED
  const barChartRef = useRef(null);
  const pieChartRef = useRef(null);
  const chartRef = useRef(null);
  const scatterChartRef = useRef(null);
  const lineChartRef = useRef(null);

  // FIXED: Initialize results with selected property
  useEffect(() => {
    const resultsWithSelection = results.map(result => ({
      ...result,
      selected: false
    }));
    setSelectedResults(resultsWithSelection);
  }, [results]);

  // Chart cleanup effect - FIXED: moved to component level
  useEffect(() => {
    return () => {
      if (barChartRef.current) {
        barChartRef.current.destroy();
      }
      if (pieChartRef.current) {
        pieChartRef.current.destroy();
      }
      if (chartRef.current) {
        chartRef.current.destroy();
      }
      if (scatterChartRef.current) {
        scatterChartRef.current.destroy();
      }
      if (lineChartRef.current) {
        lineChartRef.current.destroy();
      }
    };
  }, []);

  const handleResultToggle = (resultId) => {
    setSelectedResults(prev =>
      prev.map(result =>
        result.configId === resultId
          ? { ...result, selected: !result.selected }
          : result
      )
    );
  };

  const toggleResultDetails = (resultId) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  };

  const getFilteredResults = () => {
    let filtered = results;

    if (filterBy !== 'all') {
      if (filterBy === 'train') {
        filtered = filtered.filter(r => r.training_metrics);
      } else if (filterBy === 'evaluate') {
        filtered = filtered.filter(r => r.metrics && Object.keys(r.metrics).length > 0);
      }
    }

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'accuracy':
          const aAcc = a.metrics?.accuracy || a.metrics?.Accuracy || 0;
          const bAcc = b.metrics?.accuracy || b.metrics?.Accuracy || 0;
          return bAcc - aAcc;
        case 'algorithm':
          return (a.modelName || '').localeCompare(b.modelName || '');
        case 'timestamp':
        default:
          return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
      }
    });

    return filtered;
  };

  const getOverviewStats = () => {
    const filteredResults = getFilteredResults();
    const totalResults = filteredResults.length;
    const trainResults = filteredResults.filter(r => r.training_metrics && Object.keys(r.training_metrics).length > 0).length;
    const evalResults = filteredResults.filter(r => r.metrics && Object.keys(r.metrics).length > 0).length;

    const algorithms = [...new Set(filteredResults.map(r => r.modelName))];
    const datasets = [...new Set(filteredResults.map(r => r.datasetName))];

    const validAccuracies = filteredResults
      .map(r => r.metrics?.accuracy || r.metrics?.Accuracy)
      .filter(acc => typeof acc === 'number' && acc > 0);

    const avgAccuracy = validAccuracies.length > 0
      ? validAccuracies.reduce((sum, acc) => sum + acc, 0) / validAccuracies.length
      : 0;

    return {
      totalResults,
      trainResults,
      evalResults,
      uniqueAlgorithms: algorithms.length,
      uniqueDatasets: datasets.length,
      averageAccuracy: avgAccuracy.toFixed(3),
      algorithms,
      datasets
    };
  };

const renderDetailedMetrics = (result) => {
    const metrics = result.metrics || {};
    const trainingMetrics = result.training_metrics || {};

    // FIXED: Better metric display order and formatting
    const displayMetrics = {};
    const metricOrder = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'];

    // Process metrics in preferred order
    metricOrder.forEach(preferredKey => {
      // Check various possible key formats
      const possibleKeys = [
        preferredKey,
        preferredKey.toLowerCase(),
        preferredKey.replace('-', '_').toLowerCase(),
        preferredKey.replace(' ', '_').toLowerCase()
      ];

      for (const key of possibleKeys) {
        if (metrics[key] !== undefined) {
          displayMetrics[preferredKey] = metrics[key];
          break;
        }
      }
    });

    // Add any remaining metrics not in the standard order
    Object.entries(metrics).forEach(([key, value]) => {
      const normalizedKey = key.charAt(0).toUpperCase() + key.slice(1);
      if (!displayMetrics[normalizedKey] && !metricOrder.some(mk =>
        mk.toLowerCase() === key.toLowerCase() ||
        mk.replace('-', '_').toLowerCase() === key.toLowerCase() ||
        mk.replace(' ', '_').toLowerCase() === key.toLowerCase()
      )) {
        displayMetrics[normalizedKey] = value;
      }
    });

    return (
      <div className="detailed-metrics">
        {/* Evaluation Metrics */}
        {Object.keys(displayMetrics).length > 0 && (
          <div className="metrics-section">
            <h5>🎯 Evaluation Metrics</h5>
            <div className="metrics-grid">
              {Object.entries(displayMetrics).map(([key, value]) => (
                <div key={key} className="metric-card">
                  <div className="metric-label">{key}</div>
                  <div className="metric-value">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </div>
                  <div className="metric-bar">
                    <div
                      className="metric-fill"
                      style={{
                        width: `${typeof value === 'number' ? Math.min(value * 100, 100) : 0}%`,
                        backgroundColor: getMetricColor(key)
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Training Performance Metrics */}
        {Object.keys(trainingMetrics).length > 0 && (
          <div className="metrics-section">
            <h5>⚡ Training Performance</h5>
            <div className="training-metrics-grid">
              <div className="training-metric">
                <span className="label">Fit Time:</span>
                <span className="value">{trainingMetrics.fit_time_seconds?.toFixed(3) || result.fit_time_seconds?.toFixed(3) || 'N/A'}s</span>
              </div>
              <div className="training-metric">
                <span className="label">Memory Usage:</span>
                <span className="value">{trainingMetrics.memory_usage_mb?.toFixed(1) || result.memory_usage_mb?.toFixed(1) || 'N/A'} MB</span>
              </div>
              <div className="training-metric">
                <span className="label">Throughput:</span>
                <span className="value">{trainingMetrics.training_throughput_samples_per_sec?.toFixed(0) || result.training_throughput?.toFixed(0) || 'N/A'} samples/s</span>
              </div>
              {(trainingMetrics.training_samples_count || result.training_samples_count) && (
                <div className="training-metric">
                  <span className="label">Samples:</span>
                  <span className="value">{trainingMetrics.training_samples_count || result.training_samples_count}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Additional Info */}
        <div className="additional-info">
          <div className="info-item">
            <span className="label">Cache:</span>
            <span className={`cache-badge ${result.from_cache ? 'hit' : 'miss'}`}>
              {result.from_cache ? '🚀 CACHE HIT' : '💻 COMPUTED'}
            </span>
          </div>
          <div className="info-item">
            <span className="label">Config ID:</span>
            <span className="value">{result.configId}</span>
          </div>
          <div className="info-item">
            <span className="label">Timestamp:</span>
            <span className="value">{new Date(result.timestamp || Date.now()).toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  };

  const getMetricColor = (metricName) => {
    const colors = {
      'Accuracy': '#10b981',
      'accuracy': '#10b981',
      'Precision': '#3b82f6',
      'precision': '#3b82f6',
      'Recall': '#8b5cf6',
      'recall': '#8b5cf6',
      'F1-Score': '#f59e0b',
      'f1_score': '#f59e0b',
      'ROC AUC': '#ef4444',
      'roc_auc': '#ef4444'
    };
    return colors[metricName] || '#6b7280';
  };

  const renderOverviewTab = () => {
    const stats = getOverviewStats();
    const filteredResults = getFilteredResults();
    const resultsWithUniqueNames = createUniqueModelNames(filteredResults); // YENİ SATIR

    return (
      <div className="overview-tab">
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Total Results</h3>
            <div className="stat-value">{stats.totalResults}</div>
          </div>
          <div className="stat-card">
            <h3>Training Results</h3>
            <div className="stat-value">{stats.trainResults}</div>
          </div>
          <div className="stat-card">
            <h3>Evaluation Results</h3>
            <div className="stat-value">{stats.evalResults}</div>
          </div>
          <div className="stat-card">
            <h3>Algorithms Tested</h3>
            <div className="stat-value">{stats.uniqueAlgorithms}</div>
          </div>
          <div className="stat-card">
            <h3>Datasets Used</h3>
            <div className="stat-value">{stats.uniqueDatasets}</div>
          </div>
          <div className="stat-card">
            <h3>Average Accuracy</h3>
            <div className="stat-value">{stats.averageAccuracy}</div>
          </div>
        </div>

        {/* Detailed results section */}
        <div className="detailed-results-section">
          <div className="section-header">
            <h4>📊 Detailed Results</h4>
            <p>Click on any result to see detailed metrics and performance data</p>
          </div>

          <div className="results-list-detailed">
            {resultsWithUniqueNames.map((result, index) => ( // DEĞİŞTİRİLDİ
              <div key={result.configId || index} className="result-item-detailed">
                <div
                  className="result-header-clickable"
                  onClick={() => toggleResultDetails(result.configId)}
                >
                  <div className="result-main-info">
                    <div className="model-dataset-info">
                      <span className="model-name">{result.uniqueModelName}</span> {/* DEĞİŞTİRİLDİ */}
                      <span className="dataset-name">on {result.datasetName}</span>
                    </div>
                    <div className="quick-metrics">
                      {(result.metrics?.Accuracy || result.metrics?.accuracy) && (
                        <span className="quick-metric accuracy">
                          Acc: {(result.metrics.Accuracy || result.metrics.accuracy).toFixed(3)}
                        </span>
                      )}
                      {(result.fit_time_seconds || result.score_time_seconds) && (
                        <span className="quick-metric time">
                          Time: {(result.fit_time_seconds || result.score_time_seconds).toFixed(3)}s
                        </span>
                      )}
                      <span className={`mode-badge ${(result.metrics && Object.keys(result.metrics).length > 0) ? 'evaluate' : 'train'}`}>
                        {(result.metrics && Object.keys(result.metrics).length > 0) ? 'EVALUATE' : 'TRAIN'}
                      </span>
                    </div>
                  </div>
                  <div className="expand-indicator">
                    {expandedResults.has(result.configId) ? '▼' : '▶'}
                  </div>
                </div>

                {expandedResults.has(result.configId) && renderDetailedMetrics(result)}
              </div>
            ))}
          </div>
        </div>

        {/* Updated charts section */}
        <div className="overview-charts">
          <div className="chart-container">
            <h4>Results by Algorithm Instance</h4> {/* BAŞLIK DEĞİŞTİRİLDİ */}
            <Bar
              key={`algorithms-${stats.totalResults}`}
              ref={barChartRef}
              data={{
                labels: resultsWithUniqueNames.map(r => r.uniqueModelName), // DEĞİŞTİRİLDİ
                datasets: [{
                  label: 'Accuracy Score',
                  data: resultsWithUniqueNames.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0), // DEĞİŞTİRİLDİ
                  backgroundColor: resultsWithUniqueNames.map((r, idx) => { // DEĞİŞTİRİLDİ - Her instance için farklı renk
                    const colors = [
                      'rgba(75, 192, 192, 0.6)',
                      'rgba(54, 162, 235, 0.6)',
                      'rgba(255, 99, 132, 0.6)',
                      'rgba(255, 205, 86, 0.6)',
                      'rgba(153, 102, 255, 0.6)',
                      'rgba(255, 159, 64, 0.6)'
                    ];
                    return colors[idx % colors.length];
                  }),
                  borderColor: resultsWithUniqueNames.map((r, idx) => { // DEĞİŞTİRİLDİ
                    const colors = [
                      'rgba(75, 192, 192, 1)',
                      'rgba(54, 162, 235, 1)',
                      'rgba(255, 99, 132, 1)',
                      'rgba(255, 205, 86, 1)',
                      'rgba(153, 102, 255, 1)',
                      'rgba(255, 159, 64, 1)'
                    ];
                    return colors[idx % colors.length];
                  }),
                  borderWidth: 1
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top',
                    labels: {
                      color: getOverviewChartTheme(isDarkMode).textColor
                    }
                  },
                  title: {
                    display: true,
                    text: 'Accuracy Comparison by Model Instance',
                    color: getOverviewChartTheme(isDarkMode).textColor
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                      color: getOverviewChartTheme(isDarkMode).textColor
                    },
                    grid: {
                      color: getOverviewChartTheme(isDarkMode).gridColor
                    }
                  },
                  x: {
                    ticks: {
                      color: getOverviewChartTheme(isDarkMode).textColor,
                      maxRotation: 45 // Uzun isimleri çapraz yazma
                    },
                    grid: {
                      color: getOverviewChartTheme(isDarkMode).gridColor
                    }
                  }
                }
              }}
              height={250}
            />
          </div>

          <div className="chart-container">
            <h4>Model Performance Distribution</h4> {/* BAŞLIK DEĞİŞTİRİLDİ */}
            <Pie
              key={`datasets-${stats.totalResults}`}
              ref={pieChartRef}
              data={{
                labels: resultsWithUniqueNames.map(r => r.uniqueModelName), // DEĞİŞTİRİLDİ
                datasets: [{
                  data: resultsWithUniqueNames.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0), // DEĞİŞTİRİLDİ
                  backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)',
                    'rgba(83, 102, 255, 0.8)',
                    'rgba(255, 99, 255, 0.8)',
                    'rgba(99, 255, 132, 0.8)'
                  ],
                  borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 205, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(199, 199, 199, 1)',
                    'rgba(83, 102, 255, 1)',
                    'rgba(255, 99, 255, 1)',
                    'rgba(99, 255, 132, 1)'
                  ],
                  borderWidth: 2
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'right',
                    labels: {
                      color: getOverviewChartTheme(isDarkMode).textColor,
                      usePointStyle: true,
                      padding: 15
                    }
                  },
                  title: {
                    display: true,
                    text: 'Accuracy Share by Model Instance',
                    color: getOverviewChartTheme(isDarkMode).textColor
                  },
                  tooltip: {
                    callbacks: {
                      label: function(context) {
                        const label = context.label || '';
                        const value = context.parsed;
                        return `${label}: ${(value * 100).toFixed(1)}%`;
                      }
                    }
                  }
                }
              }}
              height={250}
            />
          </div>
        </div>
      </div>
    );
  };

  const renderDetailsTab = () => {
    const filteredResults = getFilteredResults();
    const resultsWithUniqueNames = createUniqueModelNames(filteredResults); // YENİ SATIR

    return (
      <div className="details-tab">
        <div className="results-table-container">
          <table className="results-table">
            <thead>
              <tr>
                <th>Select</th>
                <th>Model Instance</th> {/* BAŞLIK DEĞİŞTİRİLDİ */}
                <th>Dataset</th>
                <th>Mode</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Time (s)</th>
                <th>From Cache</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {resultsWithUniqueNames.map((result, index) => ( // DEĞİŞTİRİLDİ
                <tr key={result.configId || index}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedResults.find(r => r.configId === result.configId)?.selected || false}
                      onChange={() => handleResultToggle(result.configId)}
                    />
                  </td>
                  <td>{result.uniqueModelName}</td> {/* DEĞİŞTİRİLDİ */}
                  <td>{result.datasetName}</td>
                  <td>
                    <span className={`mode-badge ${(result.metrics && Object.keys(result.metrics).length > 0) ? 'evaluate' : 'train'}`}>
                      {(result.metrics && Object.keys(result.metrics).length > 0) ? 'Evaluate' : 'Train'}
                    </span>
                  </td>
                  <td>{result.metrics?.accuracy?.toFixed(3) || result.metrics?.Accuracy?.toFixed(3) || 'N/A'}</td>
                  <td>{result.metrics?.precision?.toFixed(3) || result.metrics?.Precision?.toFixed(3) || 'N/A'}</td>
                  <td>{result.metrics?.recall?.toFixed(3) || result.metrics?.Recall?.toFixed(3) || 'N/A'}</td>
                  <td>{result.metrics?.f1_score?.toFixed(3) || result.metrics?.['F1-Score']?.toFixed(3) || 'N/A'}</td>
                  <td>{(result.fit_time_seconds || result.score_time_seconds || 0).toFixed(3)}</td>
                  <td>
                    <span className={`cache-badge ${result.from_cache ? 'hit' : 'miss'}`}>
                      {result.from_cache ? 'HIT' : 'MISS'}
                    </span>
                  </td>
                  <td>
                    <button
                      className="view-details-btn"
                      onClick={() => setActiveTab('analysis')}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderChartsTab = () => {
    const filteredResults = getFilteredResults().filter(r => r.metrics && Object.keys(r.metrics).length > 0);

    if (filteredResults.length === 0) {
      return (
        <div className="no-data">
          <p>No evaluation results available for charting.</p>
          <p>Please run some model evaluations first to see performance charts.</p>
          <p>Current results appear to be training-only data.</p>
        </div>
      );
    }

    return (
      <div className="charts-tab">
        <div className="charts-grid">
          <div key={`metrics-chart-${filteredResults.length}`}>
            <MetricsChart results={filteredResults} isDarkMode={isDarkMode} />
          </div>
          <div key={`performance-chart-${filteredResults.length}`}>
            <PerformanceChart results={filteredResults} isDarkMode={isDarkMode} />
          </div>
          <div key={`roc-curve-${filteredResults.length}`}>
            <ROCCurveChart results={filteredResults} isDarkMode={isDarkMode} />
          </div>
          <div key={`confusion-matrix-${filteredResults.length}`}>
            <ConfusionMatrix results={filteredResults} isDarkMode={isDarkMode} />
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonTab = () => {
    const selectedForComparison = selectedResults.filter(r => r.selected);

    return (
      <div className="comparison-tab">
        <div className="comparison-header">
          <h4>Model Comparison ({selectedForComparison.length} selected)</h4>
          <p>Select results from the Details tab to compare them here.</p>
        </div>
        <ComparisonTable results={selectedForComparison} />
      </div>
    );
  };

  const handleOpenCodeViewer = (modelName) => {
    setSelectedModelForCode(modelName);
    setCodeViewerOpen(true);
  };

  const handleCloseCodeViewer = () => {
    setCodeViewerOpen(false);
    setSelectedModelForCode(null);
  };

  const renderAnalysisTab = () => {
    const filteredResults = getFilteredResults();

    let bestResult = filteredResults.reduce((best, current) => {
      const currentAcc = current.metrics?.accuracy || current.metrics?.Accuracy || 0;
      const bestAcc = best?.metrics?.accuracy || best?.metrics?.Accuracy || 0;
      return currentAcc > bestAcc ? current : best;
    }, null);

    if (!bestResult || (!bestResult.metrics || Object.keys(bestResult.metrics).length === 0)) {
      bestResult = filteredResults.find(r => r.training_metrics && Object.keys(r.training_metrics).length > 0);
    }
    const usedModels = [...new Set(filteredResults.map(r => r.modelName))].filter(Boolean);

    return (
      <div className="analysis-tab">
        <div className="analysis-content">
          <div className="best-result-card">
            <h4>Best Performing Model</h4>
            {bestResult ? (
              <div className="best-result-details">
                <div className="model-info">
                  <strong>{bestResult.modelName}</strong> on <strong>{bestResult.datasetName}</strong>
                </div>
                <div className="performance-metrics">
                  {/* Evaluation metrics */}
                  {bestResult.metrics && Object.entries(bestResult.metrics).map(([key, value]) => (
                    <div key={key} className="metric-item">
                      <span className="metric-label">{key}:</span>
                      <span className="metric-value">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </span>
                    </div>
                  ))}
                  {/* Training metrics if no evaluation metrics */}
                  {(!bestResult.metrics || Object.keys(bestResult.metrics).length === 0) &&
                   bestResult.training_metrics && Object.entries(bestResult.training_metrics).map(([key, value]) => (
                    <div key={key} className="metric-item">
                      <span className="metric-label">{key}:</span>
                      <span className="metric-value">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </span>
                    </div>
                  ))}
                </div>
                {bestResult.enhanced_results?.evaluation_analysis && (
                  <div className="analysis-insights">
                    <h5>Performance Analysis</h5>
                    <div className="insight-item">
                      <strong>Performance Level:</strong> {bestResult.enhanced_results.evaluation_analysis.performance_level}
                    </div>
                    {bestResult.enhanced_results.evaluation_analysis.strengths?.length > 0 && (
                      <div className="insight-item">
                        <strong>Strengths:</strong>
                        <ul>
                          {bestResult.enhanced_results.evaluation_analysis.strengths.map((strength, idx) => (
                            <li key={idx}>{strength}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <p>No results available for analysis.</p>
            )}
          </div>

          <div className="recommendations-card">
            <h4>Recommendations</h4>
            {bestResult?.recommendations?.length > 0 ? (
              <ul className="recommendations-list">
                {bestResult.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            ) : bestResult?.enhanced_results?.recommendations ? (
              <div className="enhanced-recommendations">
                <MarkdownRenderer content={bestResult.enhanced_results.recommendations} />
              </div>
            ) : (
              <p>No specific recommendations available.</p>
            )}
          </div>

          <div className="performance-trends">
            <h4>Performance Trends</h4>
              <Line
                key={`trends-${filteredResults.length}`}
                data={{
                  labels: filteredResults.map((_, idx) => `Run ${idx + 1}`),
                  datasets: [{
                    label: 'Accuracy',
                    data: filteredResults.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                  }]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'top',
                      labels: {
                        color: getOverviewChartTheme(isDarkMode).textColor
                      }
                    },
                    title: {
                      display: true,
                      text: 'Accuracy Trends Over Time',
                      color: getOverviewChartTheme(isDarkMode).textColor
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                      ticks: {
                        color: getOverviewChartTheme(isDarkMode).textColor
                      },
                      grid: {
                        color: getOverviewChartTheme(isDarkMode).gridColor
                      }
                    },
                    x: {
                      ticks: {
                        color: getOverviewChartTheme(isDarkMode).textColor
                      },
                      grid: {
                        color: getOverviewChartTheme(isDarkMode).gridColor
                      }
                    }
                  }
                }}
                height={300}
              />
          </div>

          {/* YENİ: Model Source Code Section - KAYNAK KOD BÖLÜMÜ */}
          {usedModels.length > 0 && (
            <div className="model-source-section">
              <h4>
                <span>📝</span>
                <span>Model Source Code</span>
              </h4>
              <p>
                View the complete Python implementation for the models used in your simulations.
                Click on any model to see its detailed source code with syntax highlighting.
              </p>
              <div className="model-source-buttons">
                {usedModels.map((modelName) => (
                  <button
                    key={modelName}
                    className="model-source-btn"
                    onClick={() => handleOpenCodeViewer(modelName)}
                  >
                    <span>🔍</span>
                    <span>{modelName}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* YENİ: Code Viewer Modal - KAYNAK KOD MODAL'I */}
        {codeViewerOpen && selectedModelForCode && (
          <CodeViewer
            isOpen={codeViewerOpen}
            onClose={handleCloseCodeViewer}
            modelName={selectedModelForCode}
            isDarkMode={isDarkMode}
          />
        )}
      </div>
    );
  };

  const renderSimulationTab = () => {
    const filteredResults = getFilteredResults();
    const resultsWithUniqueNames = createUniqueModelNames(filteredResults); // YENİ

    // Get unique algorithms and datasets from results
    const uniqueAlgorithms = [...new Set(resultsWithUniqueNames.map(r => r.modelName))].filter(Boolean);
    const uniqueDatasets = [...new Set(resultsWithUniqueNames.map(r => r.datasetName))].filter(Boolean);

    // Her algoritma için en iyi result'ı bul VE gerçek veriyi topla
    const algorithmResults = uniqueAlgorithms.map(algorithm => {
      const algorithmSpecificResults = resultsWithUniqueNames.filter(r => r.modelName === algorithm);
      const bestResult = algorithmSpecificResults.reduce((best, current) => {
        const currentAcc = current.metrics?.accuracy || current.metrics?.Accuracy || 0;
        const bestAcc = best?.metrics?.accuracy || best?.metrics?.Accuracy || 0;
        return currentAcc >= bestAcc ? current : best;
      }, algorithmSpecificResults[0]);

      return {
        algorithm,
        result: bestResult,
        dataset: bestResult?.datasetName || uniqueDatasets[0] || 'Unknown Dataset',
        allResults: algorithmSpecificResults // YENİ: Tüm sonuçları da ekle
      };
    });

    return (
      <div className="simulation-tab">
        <div className="simulation-header">
          <h3>Interactive ML Simulations</h3>
          <p>Watch machine learning algorithms learn in real-time with your actual experiment data!</p>
          {resultsWithUniqueNames.length > 0 && (
            <div className="simulation-stats-summary">
              <span>📊 <strong>{resultsWithUniqueNames.length}</strong> experiments</span>
              <span>🤖 <strong>{uniqueAlgorithms.length}</strong> algorithms</span>
              <span>📁 <strong>{uniqueDatasets.length}</strong> datasets</span>
              <span>🏆 Best: <strong>{Math.max(...resultsWithUniqueNames.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0)).toFixed(3)}</strong></span>
            </div>
          )}
        </div>

        {/* 1. DATASET EXPLORATION ANIMATION - Dynamic data */}
        <div className="simulation-section">
          <div className="section-header">
            <h4>Dataset Exploration Journey</h4>
            <p>Watch how we prepare and analyze data before training</p>
            {uniqueDatasets.length > 0 && (
              <p className="data-context">
                Real data from <strong>{uniqueDatasets[0]}</strong>
                {uniqueDatasets.length > 1 && ` (+${uniqueDatasets.length - 1} more datasets)`}
              </p>
            )}
          </div>

          {uniqueDatasets.length > 0 ? (
            <DatasetExplorationAnimation
              dataset={uniqueDatasets[0]}
              algorithm={uniqueAlgorithms[0] || 'Machine Learning'}
              resultsData={filteredResults.filter(r => r.datasetName === uniqueDatasets[0])} // YENİ PROP
            />
          ) : (
            <div className="no-simulation-data">
              <h5>📝 No Dataset Available</h5>
              <p>Run some experiments to see dataset exploration!</p>
              <p>Go to the main simulation panel and select a dataset, then train a model.</p>
            </div>
          )}
        </div>

        {/* 2. TRAINING ANIMATION SECTION - HER ALGORİTMA İÇİN */}
        <div className="simulation-section">
          <div className="section-header">
            <h4>Real-Time Training Animations</h4>
            <p>Experience how each algorithm learns step by step with your actual experimental data!</p>
            <p className="algorithms-context">
              Showing animations for <strong>{uniqueAlgorithms.length}</strong> different algorithm{uniqueAlgorithms.length !== 1 ? 's' : ''}:
              <strong> {uniqueAlgorithms.join(', ')}</strong>
            </p>
          </div>

          {algorithmResults.length > 0 ? (
            <div className="training-animations-container">
              {algorithmResults.map((item, index) => (
                <div key={item.algorithm} className="single-training-animation">
                  <div className="animation-title">
                    <h5>🎯 Algorithm #{index + 1}: {item.algorithm}</h5>
                    <p>
                      Dataset: <strong>{item.dataset}</strong> |
                      Best Accuracy: <strong>{(item.result?.metrics?.accuracy || item.result?.metrics?.Accuracy || 0).toFixed(3)}</strong> |
                      Status: {item.result?.from_cache ? '🚀 Cached' : '💻 Computed'} |
                      Instances: <strong>{item.allResults.length}</strong>
                    </p>
                  </div>
                  <TrainingAnimation
                    algorithm={item.algorithm}
                    dataset={item.dataset}
                    resultsData={item.allResults} // YENİ PROP - algoritma spesifik veriler
                    onComplete={() => console.log(`${item.algorithm} animation completed`)}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="no-simulation-data">
              <h5>No Results Available</h5>
              <p>Run some model training or evaluation to see simulations!</p>
              <p>Go to the main simulation panel and train a model first.</p>
            </div>
          )}
        </div>

        {/* 5. ALGORITHM RACE SECTION */}
        <div className="simulation-section">
          <div className="section-header">
            <h4>Algorithm Learning Race</h4>
            <p>Watch different algorithms compete to learn the same dataset!</p>
            {uniqueAlgorithms.length >= 2 && (
              <p className="race-context">
                Racing: <strong>{uniqueAlgorithms.slice(0, 6).join(' vs ')}</strong>
                {uniqueAlgorithms.length > 6 && ` (+${uniqueAlgorithms.length - 6} more)`}
              </p>
            )}
          </div>

          {uniqueAlgorithms.length >= 2 ? (
            <AlgorithmRace
              selectedAlgorithms={uniqueAlgorithms.slice(0, 6)} // Max 6 algorithms for performance
              dataset={uniqueDatasets[0] || 'Unknown Dataset'}
              resultsData={filteredResults} // YENİ PROP - tüm veriler
            />
          ) : uniqueAlgorithms.length === 1 ? (
            <div className="single-algorithm-notice">
              <h5>Single Algorithm Detected</h5>
              <p>Current algorithm: <strong>{uniqueAlgorithms[0]}</strong></p>
              <p>Accuracy: <strong>{Math.max(...resultsWithUniqueNames.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0)).toFixed(3)}</strong></p>
              <p>To see the Algorithm Race, train multiple different algorithms and come back here!</p>
              <p><strong>Suggested steps:</strong></p>
              <ol style={{textAlign: 'left', marginLeft: '20px'}}>
                <li>Go back to main simulation panel</li>
                <li>Train a <strong>Decision Tree</strong> model</li>
                <li>Train a <strong>SVM</strong> model</li>
                <li>Train a <strong>Neural Network</strong> model</li>
                <li>Come back to see the race!</li>
              </ol>
            </div>
          ) : (
            <div className="no-algorithms-notice">
              <h5>Ready for Algorithm Race!</h5>
              <p>Train at least 2 different algorithms to see them race against each other.</p>
              <p>Each algorithm will compete to achieve the highest accuracy fastest!</p>
            </div>
          )}
        </div>

        {/* 6. SIMULATION TIPS - Güncellenmiş verilerle */}
        <div className="simulation-tips">
          <h4>Simulation Tips</h4>
          <div className="tips-grid">
            <div className="tip-card">
              <h5>📊 Dataset Exploration</h5>
              <ul>
                <li>Real data from your {uniqueDatasets.length} dataset{uniqueDatasets.length !== 1 ? 's' : ''}</li>
                <li>Cross-validation vs train/test split comparison</li>
                <li>Actual data quality issues from your experiments</li>
                <li>Feature correlation patterns from real data</li>
              </ul>
            </div>

            <div className="tip-card">
              <h5>🎯 Training Animation</h5>
              <ul>
                <li>Each animation uses your actual model performance</li>
                <li>Training curves based on real metrics</li>
                <li>Memory usage and time data from experiments</li>
                <li>Cache status shows actual computation state</li>
              </ul>
            </div>

            <div className="tip-card">
              <h5>🧠 Neural Networks</h5>
              <ul>
                <li>Network architecture adapts to performance</li>
                <li>Real accuracy, precision, recall values</li>
                <li>Actual memory usage and throughput data</li>
                <li>Training curves from your experiments</li>
              </ul>
            </div>

            <div className="tip-card">
              <h5>🌳 Decision Trees</h5>
              <ul>
                <li>Tree growth based on real performance</li>
                <li>Actual F1-score, precision, recall metrics</li>
                <li>Real training time and memory usage</li>
                <li>Split criteria from actual feature importance</li>
              </ul>
            </div>

            <div className="tip-card">
              <h5>🏁 Algorithm Race</h5>
              <ul>
                <li>Race results based on actual performance</li>
                <li>Real training times and accuracy scores</li>
                <li>Multiple instances of same algorithm compete</li>
                <li>Cache hits vs computed results visualization</li>
              </ul>
            </div>

            <div className="tip-card">
              <h5>📊 Best Practices</h5>
              <ul>
                <li>Current data: {resultsWithUniqueNames.length} total experiments</li>
                <li>Try different hyperparameters for variety</li>
                <li>Compare cached vs computed performance</li>
                <li>Export simulation results for reports</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 7. SIMULATION STATISTICS - Güncellenmiş gerçek verilerle */}
        {filteredResults.length > 0 && (
          <div className="simulation-stats">
            <h4>📈 Your Simulation History</h4>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-value">{filteredResults.length}</span>
                <span className="stat-label">Total Experiments</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{uniqueAlgorithms.length}</span>
                <span className="stat-label">Algorithms Tried</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{uniqueDatasets.length}</span>
                <span className="stat-label">Datasets Used</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {Math.max(...filteredResults.map(r =>
                    r.metrics?.accuracy || r.metrics?.Accuracy || 0
                  )).toFixed(3)}
                </span>
                <span className="stat-label">Best Accuracy</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {(filteredResults.reduce((sum, r) => sum + (r.fit_time_seconds || 0), 0) / filteredResults.length).toFixed(2)}s
                </span>
                <span className="stat-label">Avg Training Time</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {filteredResults.filter(r => r.from_cache).length}
                </span>
                <span className="stat-label">Cache Hits</span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="results-viewer loading">
        <div className="loading-spinner"></div>
        <p>Processing simulation results...</p>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="results-viewer empty">
        <div className="empty-state">
          <h3>No Results Yet</h3>
          <p>Run some simulations to see results here!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-viewer">
      <div className="results-header">
        <div className="header-left">
          <h3>Simulation Results</h3>
          <span className="results-count">{results.length} results</span>
        </div>

        <div className="header-controls">
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Results</option>
            <option value="train">Training Only</option>
            <option value="evaluate">Evaluation Only</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="sort-select"
          >
            <option value="timestamp">Sort by Time</option>
            <option value="accuracy">Sort by Accuracy</option>
            <option value="algorithm">Sort by Algorithm</option>
          </select>

          <button
            onClick={onClearResults}
            className="clear-results-btn"
          >
            Clear Results
          </button>
        </div>
      </div>

      <div className="results-tabs">
        <button
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Overview
        </button>
        <button
          className={`tab-button ${activeTab === 'details' ? 'active' : ''}`}
          onClick={() => setActiveTab('details')}
        >
          📋 Details
        </button>
        <button
          className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
          onClick={() => setActiveTab('charts')}
        >
          📈 Charts
        </button>
        {/* YENİ TAB EKLE */}
        <button
          className={`tab-button ${activeTab === 'simulation' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulation')}
        >
          🎬 Simulation
        </button>
        <button
          className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          ⚖️ Comparison
        </button>
        <button
          className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          🔍 Analysis
        </button>
      </div>

      <div className="results-content">
        {activeTab === 'overview' && renderOverviewTab()}
        {activeTab === 'details' && renderDetailsTab()}
        {activeTab === 'charts' && renderChartsTab()}
        {activeTab === 'simulation' && renderSimulationTab()} {/* YENİ SATIR */}
        {activeTab === 'comparison' && renderComparisonTab()}
        {activeTab === 'analysis' && renderAnalysisTab()}
      </div>
    </div>
  );
};

const getOverviewChartTheme = (isDarkMode) => {
  if (isDarkMode) {
    return {
      textColor: '#e5e7eb',
      gridColor: '#4b5563',
      backgroundColor: '#1f2937'
    };
  } else {
    return {
      textColor: '#374151',
      gridColor: '#e5e7eb',
      backgroundColor: '#ffffff'
    };
  }
};

export default ResultsViewer;

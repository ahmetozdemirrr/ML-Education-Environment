// frontend/src/components/MetricsChart.js - FIXED VERSION

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

const MetricsChart = ({ results }) => {
  const chartRef = useRef(null);

  // FIXED: Chart cleanup effect with proper dependency
  useEffect(() => {
    return () => {
      // Copy ref value to variable inside effect to avoid ESLint warnings
      const currentChart = chartRef.current;
      if (currentChart) {
        currentChart.destroy();
      }
    };
  }, []);

  if (!results || results.length === 0) {
      return <div className="chart-placeholder">No metrics data available</div>;
  }

  // Filter results that have actual metrics data
  const resultsWithMetrics = results.filter(r => r.metrics && Object.keys(r.metrics).length > 0);

  if (resultsWithMetrics.length === 0) {
      return (
          <div className="chart-placeholder">
              <p>No evaluation metrics available for charting.</p>
              <p>Please run some evaluations first to see performance charts.</p>
          </div>
      );
  }

  const metricsData = resultsWithMetrics.map(result => ({
    label: `${result.modelName} (${result.datasetName})`,
    accuracy: result.metrics?.accuracy || result.metrics?.Accuracy || 0,
    precision: result.metrics?.precision || result.metrics?.Precision || 0,
    recall: result.metrics?.recall || result.metrics?.Recall || 0,
    f1_score: result.metrics?.f1_score || result.metrics?.['F1-Score'] || 0,
    roc_auc: result.metrics?.roc_auc || result.metrics?.['ROC AUC'] || 0  // BU SATIRI EKLE
  }));

  const barData = {
    labels: metricsData.map(d => d.label),
    datasets: [
      {
        label: 'Accuracy',
        data: metricsData.map(d => d.accuracy),
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      },
      {
        label: 'Precision',
        data: metricsData.map(d => d.precision),
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
      },
      {
        label: 'Recall',
        data: metricsData.map(d => d.recall),
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      },
      {
        label: 'F1-Score',
        data: metricsData.map(d => d.f1_score),
        backgroundColor: 'rgba(255, 205, 86, 0.6)',
        borderColor: 'rgba(255, 205, 86, 1)',
        borderWidth: 1
      },
      {
        label: 'ROC AUC',
        data: metricsData.map(d => d.roc_auc),
        backgroundColor: 'rgba(139, 92, 246, 0.6)',
        borderColor: 'rgba(139, 92, 246, 1)',
        borderWidth: 1
      }
    ]
  };

  const radarData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'], // ROC AUC eklendi
    datasets: metricsData.slice(0, 5).map((data, index) => ({
      label: data.label,
      data: [data.accuracy, data.precision, data.recall, data.f1_score, data.roc_auc], // ROC AUC eklendi
      backgroundColor: `rgba(${75 + index * 40}, ${192 - index * 30}, ${192 + index * 15}, 0.2)`,
      borderColor: `rgba(${75 + index * 40}, ${192 - index * 30}, ${192 + index * 15}, 1)`,
      borderWidth: 2,
      pointBackgroundColor: `rgba(${75 + index * 40}, ${192 - index * 30}, ${192 + index * 15}, 1)`
    }))
  };

  return (
    <div className="metrics-chart-container">
      <h4>Performance Metrics Comparison</h4>
      <div className="chart-tabs">
        <div className="chart-section">
          <h5>Bar Chart Comparison</h5>
          <Bar
            ref={chartRef}
            data={barData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              interaction: {
                intersect: false,
              },
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Model Performance Metrics' }
              },
              scales: {
                y: { beginAtZero: true, max: 1 }
              },
              // FIXED: Add size constraints
              aspectRatio: 2,
              layout: {
                padding: 10
              }
            }}
            height={300}
          />
        </div>

        {metricsData.length <= 5 && (
          <div className="chart-section">
            <h5>Radar Chart Comparison</h5>
            <Radar
              data={radarData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  intersect: false,
                },
                plugins: {
                  legend: { position: 'top' },
                  title: { display: true, text: 'Performance Radar' }
                },
                scales: {
                  r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                      stepSize: 0.2
                    }
                  }
                },
                // FIXED: Add size constraints
                aspectRatio: 1,
                layout: {
                  padding: 10
                }
              }}
              height={300}
            />
          </div>
        )}
      </div>
    </div>
  );
};

const PerformanceChart = ({ results }) => {
  const scatterChartRef = useRef(null);
  const lineChartRef = useRef(null);

  // FIXED: Chart cleanup effect with proper dependencies
  useEffect(() => {
    return () => {
      // Copy ref values to variables inside effect to avoid ESLint warnings
      const currentScatterChart = scatterChartRef.current;
      const currentLineChart = lineChartRef.current;

      if (currentScatterChart) {
        currentScatterChart.destroy();
      }
      if (currentLineChart) {
        currentLineChart.destroy();
      }
    };
  }, []);

  if (!results || results.length === 0) {
    return <div className="chart-placeholder">No performance data available</div>;
  }

  const performanceData = results.map((result, index) => ({
    x: result.fit_time_seconds || result.score_time_seconds || 0,
    y: result.metrics?.accuracy || result.metrics?.Accuracy || 0,
    label: result.modelName,
    dataset: result.datasetName,
    memory: result.memory_usage_mb || 0,
    index
  }));

  const scatterData = {
    datasets: [{
      label: 'Accuracy vs Training Time',
      data: performanceData,
      backgroundColor: 'rgba(255, 99, 132, 0.6)',
      borderColor: 'rgba(255, 99, 132, 1)',
    }]
  };

  const timeSeriesData = {
    labels: results.map((_, idx) => `Run ${idx + 1}`),
    datasets: [
      {
        label: 'Accuracy',
        data: results.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        yAxisID: 'y'
      },
      {
        label: 'Training Time (s)',
        data: results.map(r => r.fit_time_seconds || r.score_time_seconds || 0),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        yAxisID: 'y1'
      }
    ]
  };

  return (
    <div className="performance-chart-container">
      <h4>Performance Analysis</h4>

      <div className="chart-section">
        <h5>Accuracy vs Training Time</h5>
        <Scatter
          ref={scatterChartRef}
          data={scatterData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              intersect: false,
            },
            plugins: {
              legend: { position: 'top' },
              title: { display: true, text: 'Performance Trade-offs' },
              tooltip: {
                callbacks: {
                  label: (context) => {
                    const point = performanceData[context.dataIndex];
                    return `${point.label} (${point.dataset}): Accuracy: ${point.y.toFixed(3)}, Time: ${point.x.toFixed(3)}s`;
                  }
                }
              }
            },
            scales: {
              x: {
                title: { display: true, text: 'Training Time (seconds)' }
              },
              y: {
                title: { display: true, text: 'Accuracy' },
                beginAtZero: true,
                max: 1
              }
            },
            // FIXED: Add size constraints
            aspectRatio: 2,
            layout: {
              padding: 10
            }
          }}
          height={300}
        />
      </div>

      <div className="chart-section">
        <h5>Performance Over Time</h5>
        <Line
          ref={lineChartRef}
          data={timeSeriesData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              intersect: false,
            },
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
            },
            // FIXED: Add size constraints
            aspectRatio: 2,
            layout: {
              padding: 10
            }
          }}
          height={300}
        />
      </div>
    </div>
  );
};

const ConfusionMatrix = ({ results }) => {
  if (!results || results.length === 0) {
    return <div className="chart-placeholder">No confusion matrix data available</div>;
  }

  // Find the first result with confusion matrix data
  const resultWithMatrix = results.find(r =>
    r.enhanced_results?.visualization_data?.confusion_matrix ||
    r.plot_data?.confusion_matrix
  );

  if (!resultWithMatrix) {
    return (
      <div className="confusion-matrix-container">
        <h4>Confusion Matrix</h4>
        <div className="no-matrix-data">
          <p>Confusion matrix data not available</p>
          <p>This feature requires detailed evaluation results</p>
        </div>
      </div>
    );
  }

  const matrixData = resultWithMatrix.enhanced_results?.visualization_data?.confusion_matrix ||
                     resultWithMatrix.plot_data?.confusion_matrix;

  const matrix = matrixData.matrix;
  const labels = matrixData.labels;

  const getIntensityColor = (value, max) => {
    const intensity = value / max;
    return `rgba(59, 130, 246, ${0.1 + intensity * 0.8})`;
  };

  const maxValue = Math.max(...matrix.flat());

  return (
    <div className="confusion-matrix-container">
      <h4>Confusion Matrix</h4>
      <div className="matrix-info">
        <span className="model-info">
          {resultWithMatrix.modelName} on {resultWithMatrix.datasetName}
        </span>
      </div>

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

      <div className="matrix-stats">
        <div className="stat">
          <span className="stat-label">Total Predictions:</span>
          <span className="stat-value">{matrix.flat().reduce((a, b) => a + b, 0)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Correct Predictions:</span>
          <span className="stat-value">{matrix.reduce((sum, row, idx) => sum + row[idx], 0)}</span>
        </div>
      </div>
    </div>
  );
};

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

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const getSortedResults = () => {
    const sorted = [...results].sort((a, b) => {
      let aVal, bVal;

      switch (sortColumn) {
        case 'accuracy':
          aVal = a.metrics?.accuracy || a.metrics?.Accuracy || 0;
          bVal = b.metrics?.accuracy || b.metrics?.Accuracy || 0;
          break;
        case 'precision':
          aVal = a.metrics?.precision || a.metrics?.Precision || 0;
          bVal = b.metrics?.precision || b.metrics?.Precision || 0;
          break;
        case 'recall':
          aVal = a.metrics?.recall || a.metrics?.Recall || 0;
          bVal = b.metrics?.recall || b.metrics?.Recall || 0;
          break;
        case 'f1_score':
          aVal = a.metrics?.f1_score || a.metrics?.['F1-Score'] || 0;
          bVal = b.metrics?.f1_score || b.metrics?.['F1-Score'] || 0;
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

    return sorted;
  };

  const sortedResults = getSortedResults();
  const bestResult = sortedResults[0];

  const SortIcon = ({ column }) => {
    if (sortColumn !== column) return <span className="sort-icon">⇅</span>;
    return sortDirection === 'asc' ? <span className="sort-icon">↑</span> : <span className="sort-icon">↓</span>;
  };

  return (
    <div className="comparison-table-container">
      <div className="comparison-header">
        <h4>Model Comparison ({results.length} selected)</h4>
        {bestResult && (
          <div className="best-performer">
            <span className="best-label">Best Performer:</span>
            <span className="best-model">{bestResult.modelName}</span>
            <span className="best-accuracy">
              ({(bestResult.metrics?.accuracy || bestResult.metrics?.Accuracy || 0).toFixed(3)} accuracy)
            </span>
          </div>
        )}
      </div>

      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('model')} className="sortable">
                Model <SortIcon column="model" />
              </th>
              <th>Dataset</th>
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
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((result, index) => {
              const accuracy = result.metrics?.accuracy || result.metrics?.Accuracy || 0;
              const precision = result.metrics?.precision || result.metrics?.Precision || 0;
              const recall = result.metrics?.recall || result.metrics?.Recall || 0;
              const f1Score = result.metrics?.f1_score || result.metrics?.['F1-Score'] || 0;
              const time = result.fit_time_seconds || result.score_time_seconds || 0;
              const memory = result.memory_usage_mb || 0;

              const performanceLevel = accuracy >= 0.9 ? 'excellent' :
                                     accuracy >= 0.8 ? 'good' :
                                     accuracy >= 0.7 ? 'fair' : 'poor';

              return (
                <tr key={result.configId || index} className={index === 0 ? 'best-row' : ''}>
                  <td className="model-cell">
                    <div className="model-name">{result.modelName}</div>
                    {index === 0 && <div className="best-badge">BEST</div>}
                  </td>
                  <td>{result.datasetName}</td>
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
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="comparison-summary">
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Average Accuracy:</span>
            <span className="stat-value">
              {(sortedResults.reduce((sum, r) => sum + (r.metrics?.accuracy || r.metrics?.Accuracy || 0), 0) / sortedResults.length).toFixed(3)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Accuracy:</span>
            <span className="stat-value">
              {Math.max(...sortedResults.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0)).toFixed(3)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Fastest Training:</span>
            <span className="stat-value">
              {Math.min(...sortedResults.map(r => r.fit_time_seconds || r.score_time_seconds || Infinity)).toFixed(3)}s
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsChart;
export { PerformanceChart, ConfusionMatrix, ComparisonTable };

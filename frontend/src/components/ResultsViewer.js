// frontend/src/components/ResultsViewer.js

import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement, PointElement, LineElement } from 'chart.js';
import { Bar, Pie, Line } from 'react-chartjs-2';
import MetricsChart from './MetricsChart';
import { PerformanceChart, ConfusionMatrix, ComparisonTable } from './MetricsChart';

import './ResultsViewer.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement, PointElement, LineElement);

const ResultsViewer = ({ results = [], isLoading = false, onClearResults }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedResults, setSelectedResults] = useState([]);
  const [filterBy, setFilterBy] = useState('all'); // 'all', 'algorithm', 'dataset'
  const [sortBy, setSortBy] = useState('timestamp'); // 'timestamp', 'accuracy', 'algorithm'

  useEffect(() => {
    setSelectedResults(results);
  }, [results]);

  const handleResultToggle = (resultId) => {
    setSelectedResults(prev =>
      prev.map(result =>
        result.configId === resultId
          ? { ...result, selected: !result.selected }
          : result
      )
    );
  };

  const getFilteredResults = () => {
    let filtered = results;

    // Apply filters
    if (filterBy !== 'all') {
      // Implement specific filters based on filterBy value
      if (filterBy === 'train') {
        filtered = filtered.filter(r => r.training_metrics);
      } else if (filterBy === 'evaluate') {
        filtered = filtered.filter(r => r.metrics);
      }
    }

    // Apply sorting
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
    const trainResults = filteredResults.filter(r => r.training_metrics).length;
    const evalResults = filteredResults.filter(r => r.metrics).length;
    
    const algorithms = [...new Set(filteredResults.map(r => r.modelName))];
    const datasets = [...new Set(filteredResults.map(r => r.datasetName))];
    
    const avgAccuracy = filteredResults
      .map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0)
      .reduce((sum, acc, _, arr) => sum + acc / arr.length, 0);

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

  const renderOverviewTab = () => {
    const stats = getOverviewStats();
    const filteredResults = getFilteredResults();

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

        <div className="overview-charts">
          <div className="chart-container">
            <h4>Results by Algorithm</h4>
            <Bar
              data={{
                labels: stats.algorithms,
                datasets: [{
                  label: 'Number of Results',
                  data: stats.algorithms.map(alg => 
                    filteredResults.filter(r => r.modelName === alg).length
                  ),
                  backgroundColor: 'rgba(75, 192, 192, 0.6)',
                  borderColor: 'rgba(75, 192, 192, 1)',
                  borderWidth: 1
                }]
              }}
              options={{
                responsive: true,
                plugins: {
                  legend: { position: 'top' },
                  title: { display: true, text: 'Results Distribution by Algorithm' }
                }
              }}
            />
          </div>

          <div className="chart-container">
            <h4>Results by Dataset</h4>
            <Pie
              data={{
                labels: stats.datasets,
                datasets: [{
                  data: stats.datasets.map(dataset => 
                    filteredResults.filter(r => r.datasetName === dataset).length
                  ),
                  backgroundColor: [
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 205, 86, 0.6)',
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(153, 102, 255, 0.6)',
                    'rgba(255, 159, 64, 0.6)'
                  ]
                }]
              }}
              options={{
                responsive: true,
                plugins: {
                  legend: { position: 'top' },
                  title: { display: true, text: 'Results Distribution by Dataset' }
                }
              }}
            />
          </div>
        </div>

        <div className="recent-results">
          <h4>Recent Results</h4>
          <div className="results-list">
            {filteredResults.slice(0, 5).map((result, index) => (
              <div key={result.configId || index} className="result-item">
                <div className="result-header">
                  <span className="model-name">{result.modelName}</span>
                  <span className="dataset-name">{result.datasetName}</span>
                  <span className="timestamp">
                    {new Date(result.timestamp || Date.now()).toLocaleString()}
                  </span>
                </div>
                <div className="result-metrics">
                  {result.metrics && Object.entries(result.metrics).slice(0, 3).map(([key, value]) => (
                    <span key={key} className="metric">
                      {key}: {typeof value === 'number' ? value.toFixed(3) : value}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderDetailsTab = () => {
    const filteredResults = getFilteredResults();

    return (
      <div className="details-tab">
        <div className="results-table-container">
          <table className="results-table">
            <thead>
              <tr>
                <th>Select</th>
                <th>Model</th>
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
              {filteredResults.map((result, index) => (
                <tr key={result.configId || index}>
                  <td>
                    <input
                      type="checkbox"
                      checked={result.selected || false}
                      onChange={() => handleResultToggle(result.configId)}
                    />
                  </td>
                  <td>{result.modelName}</td>
                  <td>{result.datasetName}</td>
                  <td>
                    <span className={`mode-badge ${result.metrics ? 'evaluate' : 'train'}`}>
                      {result.metrics ? 'Evaluate' : 'Train'}
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
    const filteredResults = getFilteredResults().filter(r => r.metrics);

    if (filteredResults.length === 0) {
      return (
        <div className="no-data">
          <p>No evaluation results available for charting.</p>
        </div>
      );
    }

    return (
      <div className="charts-tab">
        <div className="charts-grid">
          <MetricsChart results={filteredResults} />
          <PerformanceChart results={filteredResults} />
          <ConfusionMatrix results={filteredResults} />
        </div>
      </div>
    );
  };

  const renderComparisonTab = () => {
    const selectedForComparison = selectedResults.filter(r => r.selected);

    return (
      <div className="comparison-tab">
        <div className="comparison-header">
          <h4>Model Comparison</h4>
          <p>Select results from the Details tab to compare them here.</p>
        </div>
        <ComparisonTable results={selectedForComparison} />
      </div>
    );
  };

  const renderAnalysisTab = () => {
    const filteredResults = getFilteredResults();
    const bestResult = filteredResults.reduce((best, current) => {
      const currentAcc = current.metrics?.accuracy || current.metrics?.Accuracy || 0;
      const bestAcc = best?.metrics?.accuracy || best?.metrics?.Accuracy || 0;
      return currentAcc > bestAcc ? current : best;
    }, null);

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
                  {bestResult.metrics && Object.entries(bestResult.metrics).map(([key, value]) => (
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
              <p>No evaluation results available.</p>
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
            ) : (
              <p>No specific recommendations available.</p>
            )}
          </div>

          <div className="performance-trends">
            <h4>Performance Trends</h4>
            <Line
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
                plugins: {
                  legend: { position: 'top' },
                  title: { display: true, text: 'Accuracy Trends Over Time' }
                },
                scales: {
                  y: { beginAtZero: true, max: 1 }
                }
              }}
            />
          </div>
        </div>
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
          Overview
        </button>
        <button 
          className={`tab-button ${activeTab === 'details' ? 'active' : ''}`}
          onClick={() => setActiveTab('details')}
        >
          Details
        </button>
        <button 
          className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
          onClick={() => setActiveTab('charts')}
        >
          Charts
        </button>
        <button 
          className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          Comparison
        </button>
        <button 
          className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          Analysis
        </button>
      </div>

      <div className="results-content">
        {activeTab === 'overview' && renderOverviewTab()}
        {activeTab === 'details' && renderDetailsTab()}
        {activeTab === 'charts' && renderChartsTab()}
        {activeTab === 'comparison' && renderComparisonTab()}
        {activeTab === 'analysis' && renderAnalysisTab()}
      </div>
    </div>
  );
};

export default ResultsViewer;

import React, { useState, useEffect, useRef } from 'react';
import { Scatter } from 'react-chartjs-2';
import DatasetService from '../../services/datasetService';

const RealDatasetExplorationAnimation = ({ dataset, algorithm, resultsData = [] }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Real dataset states
  const [realDatasetData, setRealDatasetData] = useState(null);
  const [currentDataPoints, setCurrentDataPoints] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [visualizationMethod, setVisualizationMethod] = useState('pca');
  const [validationType, setValidationType] = useState('train_test');
  const [currentFold, setCurrentFold] = useState(0);
  const [showRealStats, setShowRealStats] = useState(true);

  // Animation states
  const [animationPhase, setAnimationPhase] = useState('loading'); // loading, quality, scaling, outliers, classes, correlation, validation
  const [currentDataSubset, setCurrentDataSubset] = useState('all'); // all, train, test, outliers, etc.

  const animationRef = useRef(null);

  // Extract real statistics from results
  const getDatasetStats = () => {
    if (resultsData.length === 0) return null;

    const accuracies = resultsData.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0).filter(a => a > 0);
    const trainingTimes = resultsData.map(r => r.fit_time_seconds || 0).filter(t => t > 0);
    const memoryUsages = resultsData.map(r => r.memory_usage_mb || 0).filter(m => m > 0);

    return {
      totalExperiments: resultsData.length,
      avgAccuracy: accuracies.length > 0 ? (accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length).toFixed(3) : 'N/A',
      bestAccuracy: accuracies.length > 0 ? Math.max(...accuracies).toFixed(3) : 'N/A',
      worstAccuracy: accuracies.length > 0 ? Math.min(...accuracies).toFixed(3) : 'N/A',
      avgTrainingTime: trainingTimes.length > 0 ? (trainingTimes.reduce((sum, t) => sum + t, 0) / trainingTimes.length).toFixed(3) : 'N/A',
      avgMemoryUsage: memoryUsages.length > 0 ? (memoryUsages.reduce((sum, m) => sum + m, 0) / memoryUsages.length).toFixed(1) : 'N/A',
      uniqueModels: [...new Set(resultsData.map(r => r.modelName))],
      uniqueDatasets: [...new Set(resultsData.map(r => r.datasetName))]
    };
  };

  const stats = getDatasetStats();

  const steps = [
    {
      name: 'Real Data Loading',
      description: realDatasetData ?
        `Loading ${realDatasetData.dataset_name} with ${realDatasetData.n_samples} samples and ${realDatasetData.original_features} features` :
        'Loading real dataset from file system...',
      phase: 'loading'
    },
    {
      name: 'Data Quality Check',
      description: realDatasetData ?
        `Checking ${realDatasetData.n_samples} samples for missing values and data integrity...` :
        'Analyzing data quality and consistency...',
      phase: 'quality'
    },
    {
      name: 'Feature Engineering',
      description: realDatasetData ?
        `Standardizing ${realDatasetData.original_features} features using ${visualizationMethod.toUpperCase()} for dimensionality reduction` :
        'Applying feature scaling and dimensionality reduction...',
      phase: 'scaling'
    },
    {
      name: 'Outlier Analysis',
      description: 'Identifying unusual data points using statistical methods...',
      phase: 'outliers'
    },
    {
      name: 'Class Distribution',
      description: realDatasetData ?
        `Analyzing ${realDatasetData.n_classes} classes: ${realDatasetData.class_names.join(', ')}` :
        'Examining target variable distribution...',
      phase: 'classes'
    },
    {
      name: 'Feature Correlation',
      description: `Computing relationships between ${realDatasetData?.original_features || 'multiple'} original features...`,
      phase: 'correlation'
    },
    {
      name: validationType === 'cross_validation' ? 'Cross Validation Setup' : 'Train/Test Split',
      description: validationType === 'cross_validation' ?
        'Setting up 5-fold cross validation for robust evaluation...' :
        `Splitting ${realDatasetData?.n_samples || 'data'} samples into training (80%) and test (20%) sets...`,
      phase: 'validation'
    }
  ];

  // Load real dataset data
  useEffect(() => {
    const loadRealDataset = async () => {
      if (!dataset) return;

      setIsLoading(true);
      setError(null);

      try {
        // Convert dataset name to filename
        const datasetFilename = DatasetService.getDatasetNameFromPath(dataset);

        // Load dataset visualization data
        const visualizationData = await DatasetService.getDatasetVisualization(datasetFilename, visualizationMethod);
        setRealDatasetData(visualizationData);

        // Load dataset info
        const info = await DatasetService.getDatasetInfo(datasetFilename);
        setDatasetInfo(info);

        // Initialize with all points
        setCurrentDataPoints(visualizationData.all_points);

      } catch (err) {
        console.error('Error loading real dataset:', err);
        setError(`Failed to load dataset: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    };

    loadRealDataset();
  }, [dataset, visualizationMethod]);

  // Update data points based on current step
  useEffect(() => {
    if (!realDatasetData) return;

    const updateDataPoints = () => {
      let points = [...realDatasetData.all_points];

      switch(currentStep) {
        case 0: // Raw Data Loading
          points = points.map(p => ({
            ...p,
            color: '#94a3b8',
            size: 3,
            displayLabel: 'raw_data'
          }));
          break;

        case 1: // Data Quality Check
          points = points.map((p, idx) => {
            if (idx % 50 === 0) { // Simulate missing values
              return { ...p, color: '#ef4444', size: 8, displayLabel: 'missing_value' };
            } else if (idx % 30 === 0) { // Simulate duplicates
              return { ...p, color: '#f59e0b', size: 6, displayLabel: 'duplicate' };
            } else {
              return { ...p, color: '#22c55e', size: 4, displayLabel: 'clean_data' };
            }
          });
          break;

        case 2: // Feature Engineering
          points = points.map(p => ({
            ...p,
            color: '#3b82f6',
            size: 4,
            displayLabel: `scaled_${visualizationMethod}`
          }));
          break;

        case 3: // Outlier Analysis
          const xValues = points.map(p => p.x);
          const yValues = points.map(p => p.y);
          const xQ1 = quantile(xValues, 0.25);
          const xQ3 = quantile(xValues, 0.75);
          const yQ1 = quantile(yValues, 0.25);
          const yQ3 = quantile(yValues, 0.75);
          const xIQR = xQ3 - xQ1;
          const yIQR = yQ3 - yQ1;

          points = points.map(p => {
            const isOutlierX = p.x < (xQ1 - 1.5 * xIQR) || p.x > (xQ3 + 1.5 * xIQR);
            const isOutlierY = p.y < (yQ1 - 1.5 * yIQR) || p.y > (yQ3 + 1.5 * yIQR);
            const isOutlier = isOutlierX || isOutlierY;

            return {
              ...p,
              color: isOutlier ? '#ef4444' : '#3b82f6',
              size: isOutlier ? 8 : 4,
              displayLabel: isOutlier ? 'outlier' : 'normal'
            };
          });
          break;

        case 4: // Class Distribution
          points = points.map(p => ({
            ...p,
            color: p.color, // Use original class colors
            size: 5,
            displayLabel: p.class_name
          }));
          break;

        case 5: // Feature Correlation
          // Simulate correlation visualization by modifying point appearance
          points = points.map(p => {
            const correlation = Math.abs(p.x * p.y) / Math.max(Math.abs(p.x), Math.abs(p.y), 1);
            const isHighCorr = correlation > 0.6;
            const isMedCorr = correlation > 0.3;

            return {
              ...p,
              color: isHighCorr ? '#22c55e' : isMedCorr ? '#f59e0b' : '#ef4444',
              size: isHighCorr ? 6 : isMedCorr ? 5 : 4,
              displayLabel: isHighCorr ? 'high_corr' : isMedCorr ? 'med_corr' : 'low_corr'
            };
          });
          break;

        case 6: // Validation Split
          if (validationType === 'cross_validation') {
            const foldSize = Math.floor(points.length / 5);
            points = points.map((p, idx) => {
              const foldIndex = Math.floor(idx / foldSize);
              const isTestFold = foldIndex === currentFold;

              return {
                ...p,
                color: isTestFold ? '#ef4444' : '#22c55e',
                size: isTestFold ? 6 : 4,
                displayLabel: isTestFold ? `test_fold_${currentFold + 1}` : 'train_fold'
              };
            });
          } else {
            // Traditional train/test split - use the actual split from backend
            points = realDatasetData.train_points.map(p => ({
              ...p,
              color: '#22c55e',
              size: 4,
              displayLabel: 'train_set'
            })).concat(realDatasetData.test_points.map(p => ({
              ...p,
              color: '#ef4444',
              size: 6,
              displayLabel: 'test_set'
            })));
          }
          break;

        default:
          break;
      }

      setCurrentDataPoints(points);
    };

    updateDataPoints();
  }, [currentStep, realDatasetData, currentFold, validationType]);

  // Quantile calculation helper
  const quantile = (arr, q) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
      return sorted[base];
    }
  };

  const startAnimation = () => {
    if (!realDatasetData) {
      setError('Please wait for dataset to load before starting animation');
      return;
    }

    setIsPlaying(true);
    setCurrentStep(0);
    setCurrentFold(0);

    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          // If it's cross validation and on the last step, cycle through folds
          if (prev === steps.length - 1 && validationType === 'cross_validation') {
            setCurrentFold(prevFold => {
              if (prevFold >= 4) { // 5 folds (0-4)
                setIsPlaying(false);
                clearInterval(interval);
                return 0;
              }
              return prevFold + 1;
            });
            return prev; // Stay on the same step but change fold
          } else {
            setIsPlaying(false);
            clearInterval(interval);
            return prev;
          }
        }
        return prev + 1;
      });
    }, 3000); // 3 seconds per step

    animationRef.current = interval;
  };

  // Stop animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    };
  }, []);

  const chartData = {
    datasets: [{
      label: 'Real Dataset Points',
      data: currentDataPoints,
      backgroundColor: currentDataPoints.map(p => p.color),
      borderColor: currentDataPoints.map(p => p.color),
      pointRadius: currentDataPoints.map(p => p.size || 4),
      pointHoverRadius: currentDataPoints.map(p => (p.size || 4) + 2),
      showLine: false
    }]
  };

  const bounds = realDatasetData?.bounds || { x_min: -3, x_max: 3, y_min: -3, y_max: 3 };
  const padding = 0.1;
  const xRange = bounds.x_max - bounds.x_min;
  const yRange = bounds.y_max - bounds.y_min;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: `${steps[currentStep]?.name} - ${realDatasetData?.dataset_name || dataset}${validationType === 'cross_validation' && currentStep === steps.length - 1 ? ` (Fold ${currentFold + 1}/5)` : ''}`,
        font: { size: 16, weight: 'bold' },
        color: '#8b4513'
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = currentDataPoints[context.dataIndex];
            return `${point.class_name || point.displayLabel}: (${point.x.toFixed(3)}, ${point.y.toFixed(3)})`;
          }
        }
      }
    },
    scales: {
      x: {
        min: bounds.x_min - (xRange * padding),
        max: bounds.x_max + (xRange * padding),
        title: {
          display: true,
          text: visualizationMethod === 'pca' ? 'Principal Component 1' : visualizationMethod === 'tsne' ? 't-SNE Dimension 1' : 'Feature 1',
          color: '#8b4513'
        },
        ticks: { color: '#8b4513' },
        grid: { color: 'rgba(139, 69, 19, 0.2)' }
      },
      y: {
        min: bounds.y_min - (yRange * padding),
        max: bounds.y_max + (yRange * padding),
        title: {
          display: true,
          text: visualizationMethod === 'pca' ? 'Principal Component 2' : visualizationMethod === 'tsne' ? 't-SNE Dimension 2' : 'Feature 2',
          color: '#8b4513'
        },
        ticks: { color: '#8b4513' },
        grid: { color: 'rgba(139, 69, 19, 0.2)' }
      }
    }
  };

  const getLegendForCurrentStep = () => {
    switch(currentStep) {
      case 1:
        return (
          <div className="step-legend">
            <span style={{color: '#22c55e'}}>🟢 Clean Data</span> |
            <span style={{color: '#f59e0b'}}> 🟡 Duplicates</span> |
            <span style={{color: '#ef4444'}}> 🔴 Missing Values</span>
          </div>
        );
      case 3:
        return (
          <div className="step-legend">
            <span style={{color: '#3b82f6'}}>🔵 Normal Points</span> |
            <span style={{color: '#ef4444'}}> 🔴 Outliers (IQR Method)</span>
          </div>
        );
      case 4:
        if (realDatasetData) {
          return (
            <div className="step-legend">
              {realDatasetData.class_names.map((className, idx) => (
                <span key={idx} style={{color: realDatasetData.colors[idx]}}>
                  ⬤ {className}
                  {idx < realDatasetData.class_names.length - 1 ? ' | ' : ''}
                </span>
              ))}
            </div>
          );
        }
        return null;
      case 5:
        return (
          <div className="step-legend">
            <span style={{color: '#22c55e'}}>🟢 High Correlation</span> |
            <span style={{color: '#f59e0b'}}> 🟡 Medium Correlation</span> |
            <span style={{color: '#ef4444'}}> 🔴 Low Correlation</span>
          </div>
        );
      case 6:
        if (validationType === 'cross_validation') {
          return (
            <div className="step-legend">
              <span style={{color: '#22c55e'}}>🟢 Training Folds (4/5)</span> |
              <span style={{color: '#ef4444'}}> 🔴 Validation Fold ({currentFold + 1}/5)</span>
            </div>
          );
        } else {
          return (
            <div className="step-legend">
              <span style={{color: '#22c55e'}}>🟢 Training Set ({realDatasetData?.train_points?.length || 'N/A'})</span> |
              <span style={{color: '#ef4444'}}> 🔴 Test Set ({realDatasetData?.test_points?.length || 'N/A'})</span>
            </div>
          );
        }
      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <div className="dataset-exploration-container">
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <h3>🔄 Loading Real Dataset...</h3>
          <p>Fetching {dataset} from file system...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dataset-exploration-container">
        <div className="error-state">
          <h3>❌ Error Loading Dataset</h3>
          <p>{error}</p>
          <p>Available datasets: {datasetInfo ? 'Loaded successfully' : 'Could not load info'}</p>
        </div>
      </div>
    );
  }

  if (!realDatasetData) {
    return (
      <div className="dataset-exploration-container">
        <div className="no-data-state">
          <h3>📊 Dataset Exploration</h3>
          <p>No dataset data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dataset-exploration-container">
      <div className="exploration-header">
        <h3>📊 Real Dataset Exploration Journey</h3>
        <p>Exploring <strong>{realDatasetData.dataset_name}</strong> with <strong>{algorithm}</strong></p>
        <p className="dataset-meta">
          {realDatasetData.n_samples} samples • {realDatasetData.original_features} features • {realDatasetData.n_classes} classes
        </p>
      </div>

      {/* Real Dataset Statistics */}
      {(stats || realDatasetData) && showRealStats && (
        <div className="real-dataset-stats">
          <div className="stats-header">
            <h4>📈 Dataset Statistics</h4>
            <button
              className="toggle-stats-btn"
              onClick={() => setShowRealStats(!showRealStats)}
            >
              {showRealStats ? '🙈 Hide' : '👁️ Show'}
            </button>
          </div>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Total Samples:</span>
              <span className="stat-value">{realDatasetData.n_samples}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Features:</span>
              <span className="stat-value">{realDatasetData.original_features}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Classes:</span>
              <span className="stat-value">{realDatasetData.n_classes}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Method:</span>
              <span className="stat-value">{realDatasetData.method_used.toUpperCase()}</span>
            </div>
            {stats && (
              <>
                <div className="stat-item">
                  <span className="stat-label">Experiments:</span>
                  <span className="stat-value">{stats.totalExperiments}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Best Accuracy:</span>
                  <span className="stat-value">{stats.bestAccuracy}</span>
                </div>
              </>
            )}
          </div>
          <div className="class-names">
            <strong>Classes:</strong> {realDatasetData.class_names.join(', ')}
          </div>
          <div className="feature-names">
            <strong>Key Features:</strong> {realDatasetData.feature_names.slice(0, 5).join(', ')}
            {realDatasetData.feature_names.length > 5 && ` ... (+${realDatasetData.feature_names.length - 5} more)`}
          </div>
        </div>
      )}

      <div className="exploration-controls">
        <button onClick={startAnimation} disabled={isPlaying || isLoading} className="start-btn">
          {isPlaying ? '🔄 Exploring...' : '🚀 Start Real Data Exploration'}
        </button>

        <div className="method-selector">
          <label>Visualization Method:</label>
          <select
            value={visualizationMethod}
            onChange={(e) => setVisualizationMethod(e.target.value)}
            disabled={isPlaying}
          >
            <option value="pca">PCA (Linear)</option>
            <option value="tsne">t-SNE (Non-linear)</option>
          </select>
        </div>

        <div className="validation-toggle">
          <label>
            <input
              type="radio"
              value="train_test"
              checked={validationType === 'train_test'}
              onChange={(e) => setValidationType(e.target.value)}
              disabled={isPlaying}
            />
            Train/Test Split
          </label>
          <label>
            <input
              type="radio"
              value="cross_validation"
              checked={validationType === 'cross_validation'}
              onChange={(e) => setValidationType(e.target.value)}
              disabled={isPlaying}
            />
            Cross Validation
          </label>
        </div>

        <div className="step-indicator">
          Step {currentStep + 1} of {steps.length}: {steps[currentStep]?.name}
        </div>
      </div>

      <div className="exploration-description">
        <p>{steps[currentStep]?.description}</p>
        {getLegendForCurrentStep()}
      </div>

      <div className="chart-container" style={{ height: '450px' }}>
        <Scatter data={chartData} options={chartOptions} />
      </div>

      <div className="progress-steps">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`progress-step ${index <= currentStep ? 'completed' : ''} ${index === currentStep ? 'active' : ''}`}
          >
            <div className="step-number">{index + 1}</div>
            <div className="step-name">{step.name}</div>
            {index === steps.length - 1 && validationType === 'cross_validation' && currentStep === index && (
              <div className="fold-indicator">Fold {currentFold + 1}/5</div>
            )}
          </div>
        ))}
      </div>

      <div className="data-summary">
        <h4>🔍 Current Data View</h4>
        <div className="summary-stats">
          <span>Visible Points: <strong>{currentDataPoints.length}</strong></span>
          <span>Current Phase: <strong>{steps[currentStep]?.phase || 'N/A'}</strong></span>
          <span>Data Source: <strong>Real {dataset}</strong></span>
        </div>
      </div>

      <style jsx>{`
        .dataset-exploration-container {
          background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: #8b4513;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .loading-state, .error-state, .no-data-state {
          text-align: center;
          padding: 40px;
        }

        .loading-spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(139, 69, 19, 0.3);
          border-left-color: #8b4513;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .exploration-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .exploration-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.4rem;
          font-weight: 700;
        }

        .dataset-meta {
          font-size: 0.9rem;
          color: #8b4513;
          opacity: 0.8;
          margin: 4px 0;
        }

        .real-dataset-stats {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 20px;
          border: 2px solid rgba(139, 69, 19, 0.3);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .stats-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .stats-header h4 {
          margin: 0;
          color: #8b4513;
          font-size: 1.1rem;
        }

        .toggle-stats-btn {
          padding: 4px 8px;
          border: none;
          border-radius: 6px;
          background: rgba(139, 69, 19, 0.1);
          color: #8b4513;
          cursor: pointer;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px;
          margin-bottom: 12px;
        }

        .stat-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: rgba(139, 69, 19, 0.05);
          border-radius: 6px;
          border-left: 3px solid #8b4513;
        }

        .stat-label {
          font-size: 0.9rem;
          color: #8b4513;
        }

        .stat-value {
          font-weight: 700;
          color: #d2691e;
          font-family: monospace;
        }

        .class-names, .feature-names {
          text-align: center;
          font-size: 0.9rem;
          color: #8b4513;
          padding: 8px;
          background: rgba(139, 69, 19, 0.05);
          border-radius: 6px;
          margin-bottom: 8px;
        }

        .exploration-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 16px;
        }

        .start-btn {
          padding: 12px 24px;
          border: none;
          border-radius: 8px;
          background: rgba(139, 69, 19, 0.8);
          color: white;
          cursor: pointer;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.3s ease;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .start-btn:hover:not(:disabled) {
          background: rgba(139, 69, 19, 1);
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .start-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .method-selector {
          display: flex;
          flex-direction: column;
          gap: 4px;
          background: rgba(255, 255, 255, 0.8);
          padding: 8px 12px;
          border-radius: 8px;
        }

        .method-selector label {
          font-size: 0.8rem;
          font-weight: 600;
          color: #8b4513;
        }

        .method-selector select {
          border: 1px solid rgba(139, 69, 19, 0.3);
          border-radius: 4px;
          padding: 4px;
          font-size: 0.9rem;
        }

        .validation-toggle {
          display: flex;
          gap: 16px;
          background: rgba(255, 255, 255, 0.8);
          padding: 8px 12px;
          border-radius: 8px;
        }

        .validation-toggle label {
          display: flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
          font-size: 0.9rem;
          font-weight: 500;
        }

        .step-indicator {
          background: rgba(255, 255, 255, 0.9);
          padding: 10px 16px;
          border-radius: 20px;
          font-weight: 600;
          font-size: 0.9rem;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .exploration-description {
          background: rgba(255, 255, 255, 0.8);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
          text-align: center;
          border-left: 4px solid #8b4513;
        }

        .exploration-description p {
          margin: 0 0 8px 0;
          font-size: 1rem;
          line-height: 1.5;
        }

        .step-legend {
          font-size: 0.9rem;
          font-weight: 500;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 20px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .progress-steps {
          display: flex;
          justify-content: space-between;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 8px;
        }

        .progress-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 12px 8px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.4);
          transition: all 0.3s ease;
          flex: 1;
          min-width: 120px;
          border: 2px solid transparent;
        }

        .progress-step.completed {
          background: rgba(34, 197, 94, 0.3);
          color: #15803d;
          border-color: #22c55e;
        }

        .progress-step.active {
          background: rgba(59, 130, 246, 0.3);
          color: #1d4ed8;
          transform: scale(1.05);
          border-color: #3b82f6;
          box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
        }

        .step-number {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: currentColor;
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: bold;
          margin-bottom: 6px;
        }

        .step-name {
          font-size: 0.8rem;
          text-align: center;
          font-weight: 600;
          line-height: 1.3;
        }

        .fold-indicator {
          font-size: 0.7rem;
          background: rgba(59, 130, 246, 0.8);
          color: white;
          padding: 2px 6px;
          border-radius: 10px;
          margin-top: 4px;
        }

        .data-summary {
          background: rgba(255, 255, 255, 0.7);
          border-radius: 8px;
          padding: 16px;
          text-align: center;
        }

        .data-summary h4 {
          margin: 0 0 12px 0;
          color: #8b4513;
        }

        .summary-stats {
          display: flex;
          justify-content: center;
          gap: 24px;
          flex-wrap: wrap;
          font-size: 0.9rem;
        }

        .summary-stats span {
          color: #8b4513;
        }

        @media (max-width: 768px) {
          .exploration-controls {
            flex-direction: column;
            align-items: stretch;
          }

          .validation-toggle {
            justify-content: center;
          }

          .progress-steps {
            flex-direction: column;
          }

          .progress-step {
            flex-direction: row;
            gap: 12px;
            justify-content: flex-start;
          }

          .summary-stats {
            flex-direction: column;
            gap: 8px;
          }

          .stats-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default RealDatasetExplorationAnimation;

import React, { useState, useEffect, useRef } from 'react';
import { Scatter } from 'react-chartjs-2';

const DatasetExplorationAnimation = ({ dataset, algorithm, resultsData = [] }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dataPoints, setDataPoints] = useState([]);
  const [validationType, setValidationType] = useState('train_test'); // 'train_test' or 'cross_validation'
  const [currentFold, setCurrentFold] = useState(0);
  const [showRealStats, setShowRealStats] = useState(true);

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
      name: 'Raw Data Loading',
      description: stats ?
        `Loading ${dataset} dataset with ${stats.totalExperiments} experiments from ${stats.uniqueModels.length} different algorithms` :
        'Loading dataset points from file...',
      phase: 'data_loading'
    },
    {
      name: 'Data Quality Check',
      description: 'Checking for missing values, duplicates, and data integrity...',
      phase: 'quality_check'
    },
    {
      name: 'Feature Scaling',
      description: stats ?
        `Normalizing features across ${stats.uniqueDatasets.length} dataset(s) using StandardScaler` :
        'Normalizing feature values using StandardScaler...',
      phase: 'scaling'
    },
    {
      name: 'Outlier Detection',
      description: 'Identifying unusual data points using IQR method...',
      phase: 'outlier_detection'
    },
    {
      name: 'Class Distribution Analysis',
      description: 'Analyzing target variable distribution and class balance...',
      phase: 'class_analysis'
    },
    {
      name: 'Feature Correlation',
      description: 'Computing correlation matrix and feature importance...',
      phase: 'correlation'
    },
    {
      name: validationType === 'cross_validation' ? 'Cross Validation Setup' : 'Train/Test Split',
      description: validationType === 'cross_validation' ?
        'Setting up 5-fold cross validation for robust model evaluation...' :
        'Splitting data into 80% training and 20% test sets...',
      phase: 'validation_setup'
    }
  ];

  // Generate synthetic data based on dataset type and step
  const generateDataPoints = (step, fold = 0) => {
    const points = [];
    const numPoints = 300;

    for (let i = 0; i < numPoints; i++) {
      let x, y, color, size, label;

      switch(step) {
        case 0: // Raw Data Loading
          x = Math.random() * 12 - 6;
          y = Math.random() * 12 - 6;
          color = '#94a3b8';
          size = 3;
          label = 'raw';
          break;

        case 1: // Data Quality Check
          if (i % 50 === 0) { // Missing values
            x = Math.random() * 12 - 6;
            y = Math.random() * 12 - 6;
            color = '#ef4444';
            size = 6;
            label = 'missing';
          } else if (i % 30 === 0) { // Duplicates
            x = Math.random() * 12 - 6;
            y = Math.random() * 12 - 6;
            color = '#f59e0b';
            size = 5;
            label = 'duplicate';
          } else {
            x = Math.random() * 12 - 6;
            y = Math.random() * 12 - 6;
            color = '#22c55e';
            size = 3;
            label = 'clean';
          }
          break;

        case 2: // Feature Scaling
          x = (Math.random() - 0.5) * 6; // Normalized to [-3, 3]
          y = (Math.random() - 0.5) * 6;
          color = '#3b82f6';
          size = 4;
          label = 'scaled';
          break;

        case 3: // Outlier Detection
          if (i < 15) { // Outliers
            x = Math.random() > 0.5 ? Math.random() * 4 + 4 : Math.random() * 4 - 8;
            y = Math.random() > 0.5 ? Math.random() * 4 + 4 : Math.random() * 4 - 8;
            color = '#ef4444';
            size = 8;
            label = 'outlier';
          } else {
            x = (Math.random() - 0.5) * 4;
            y = (Math.random() - 0.5) * 4;
            color = '#3b82f6';
            size = 4;
            label = 'normal';
          }
          break;

        case 4: // Class Distribution
          const classNum = i % 3;
          const classOffsets = [[-2, -1], [0, 2], [2, -1]];
          x = Math.random() * 2 - 1 + classOffsets[classNum][0];
          y = Math.random() * 2 - 1 + classOffsets[classNum][1];
          color = ['#ef4444', '#22c55e', '#8b5cf6'][classNum];
          size = 5;
          label = `class_${classNum}`;
          break;

        case 5: // Feature Correlation
          x = Math.random() * 4 - 2;
          y = x * 0.8 + Math.random() * 0.8 - 0.4; // Strong positive correlation
          const correlationStrength = Math.abs(y / x);
          color = correlationStrength > 0.6 ? '#22c55e' : correlationStrength > 0.3 ? '#f59e0b' : '#ef4444';
          size = 4;
          label = 'correlated';
          break;

        case 6: // Validation Split
          if (validationType === 'cross_validation') {
            // 5-fold cross validation
            const foldSize = Math.floor(numPoints / 5);
            const currentFoldStart = fold * foldSize;
            const currentFoldEnd = (fold + 1) * foldSize;

            x = Math.random() * 4 - 2;
            y = Math.random() * 4 - 2;

            if (i >= currentFoldStart && i < currentFoldEnd) {
              color = '#ef4444'; // Test fold
              size = 6;
              label = 'test_fold';
            } else {
              color = '#22c55e'; // Train folds
              size = 4;
              label = 'train_fold';
            }
          } else {
            // Traditional train/test split
            const isTrain = i < numPoints * 0.8;
            x = Math.random() * 4 - 2;
            y = Math.random() * 4 - 2;
            color = isTrain ? '#22c55e' : '#ef4444';
            size = isTrain ? 4 : 6;
            label = isTrain ? 'train' : 'test';
          }
          break;

        default:
          x = y = 0;
          color = '#94a3b8';
          size = 4;
          label = 'default';
      }

      points.push({ x, y, color, size, label, id: i });
    }

    return points;
  };

  useEffect(() => {
    setDataPoints(generateDataPoints(currentStep, currentFold));
  }, [currentStep, currentFold, validationType]);

  const startAnimation = () => {
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
    }, 2500);
  };

  const chartData = {
    datasets: [{
      label: 'Data Points',
      data: dataPoints,
      backgroundColor: dataPoints.map(p => p.color),
      borderColor: dataPoints.map(p => p.color),
      pointRadius: dataPoints.map(p => p.size),
      pointHoverRadius: dataPoints.map(p => p.size + 2),
      showLine: false
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: `${steps[currentStep]?.name} - ${dataset}${validationType === 'cross_validation' && currentStep === steps.length - 1 ? ` (Fold ${currentFold + 1}/5)` : ''}`,
        font: { size: 16, weight: 'bold' },
        color: '#8b4513'
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = dataPoints[context.dataIndex];
            return `${point.label}: (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
          }
        }
      }
    },
    scales: {
      x: {
        min: -8,
        max: 8,
        title: { display: true, text: 'Feature 1 (Standardized)', color: '#8b4513' },
        ticks: { color: '#8b4513' },
        grid: { color: 'rgba(139, 69, 19, 0.2)' }
      },
      y: {
        min: -8,
        max: 8,
        title: { display: true, text: 'Feature 2 (Standardized)', color: '#8b4513' },
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
        return (
          <div className="step-legend">
            <span style={{color: '#ef4444'}}>🔴 Class A</span> |
            <span style={{color: '#22c55e'}}> 🟢 Class B</span> |
            <span style={{color: '#8b5cf6'}}> 🟣 Class C</span>
          </div>
        );
      case 5:
        return (
          <div className="step-legend">
            <span style={{color: '#22c55e'}}>🟢 Strong Correlation</span> |
            <span style={{color: '#f59e0b'}}> 🟡 Moderate Correlation</span> |
            <span style={{color: '#ef4444'}}> 🔴 Weak Correlation</span>
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
              <span style={{color: '#22c55e'}}>🟢 Training Set (80%)</span> |
              <span style={{color: '#ef4444'}}> 🔴 Test Set (20%)</span>
            </div>
          );
        }
      default:
        return null;
    }
  };

  return (
    <div className="dataset-exploration-container">
      <div className="exploration-header">
        <h3>📊 Dataset Exploration Journey</h3>
        <p>Watch how we prepare <strong>{dataset}</strong> data for <strong>{algorithm}</strong></p>
      </div>

      {/* Real Dataset Statistics */}
      {stats && showRealStats && (
        <div className="real-dataset-stats">
          <div className="stats-header">
            <h4>📈 Real Experimental Statistics</h4>
            <button
              className="toggle-stats-btn"
              onClick={() => setShowRealStats(!showRealStats)}
            >
              {showRealStats ? '🙈 Hide' : '👁️ Show'}
            </button>
          </div>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Total Experiments:</span>
              <span className="stat-value">{stats.totalExperiments}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Algorithms Used:</span>
              <span className="stat-value">{stats.uniqueModels.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Average Accuracy:</span>
              <span className="stat-value">{stats.avgAccuracy}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Best Accuracy:</span>
              <span className="stat-value">{stats.bestAccuracy}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Training Time:</span>
              <span className="stat-value">{stats.avgTrainingTime}s</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Memory Usage:</span>
              <span className="stat-value">{stats.avgMemoryUsage} MB</span>
            </div>
          </div>
          <div className="algorithms-used">
            <strong>Algorithms:</strong> {stats.uniqueModels.join(', ')}
          </div>
        </div>
      )}

      <div className="exploration-controls">
        <button onClick={startAnimation} disabled={isPlaying} className="start-btn">
          {isPlaying ? '🔄 Exploring...' : '🚀 Start Exploration'}
        </button>

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
        <h4>🔍 Current Data Summary</h4>
        <div className="summary-stats">
          <span>Total Points: <strong>{dataPoints.length}</strong></span>
          <span>Unique Labels: <strong>{[...new Set(dataPoints.map(p => p.label))].length}</strong></span>
          <span>Phase: <strong>{steps[currentStep]?.phase || 'N/A'}</strong></span>
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

        .exploration-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .exploration-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.4rem;
          font-weight: 700;
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
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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

        .algorithms-used {
          text-align: center;
          font-size: 0.9rem;
          color: #8b4513;
          padding: 8px;
          background: rgba(139, 69, 19, 0.05);
          border-radius: 6px;
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

export default DatasetExplorationAnimation;

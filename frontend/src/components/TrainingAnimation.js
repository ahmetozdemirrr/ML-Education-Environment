// frontend/src/components/TrainingAnimation.js

import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrainingAnimation = ({ algorithm, dataset, onComplete }) => {
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500); // ms per epoch
  const intervalRef = useRef(null);

  // Simulated training data for different algorithms
  const simulateTraining = (algorithm, totalEpochs = 50) => {
    const history = [];

    switch(algorithm) {
      case 'Decision Tree':
        // Simulate tree growth
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.95, 0.5 + (i / totalEpochs) * 0.4 + Math.random() * 0.05),
            loss: Math.max(0.05, 1.0 - (i / totalEpochs) * 0.8 + Math.random() * 0.1),
            treeDepth: Math.min(10, Math.floor(i / 5) + 1),
            nodes: Math.min(100, i * 2 + 1)
          });
        }
        break;

      case 'SVM':
      case 'Support Vector Machine':
        // Simulate margin optimization
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.92, 0.6 + (i / totalEpochs) * 0.3 + Math.random() * 0.02),
            loss: Math.max(0.1, 1.2 - (i / totalEpochs) * 0.9 + Math.random() * 0.05),
            margin: (i / totalEpochs) * 0.8 + 0.1,
            supportVectors: Math.min(20, Math.floor(i / 3) + 3)
          });
        }
        break;

      case 'Artificial Neural Network':
      case 'Neural Network':
        // Simulate neural network training
        for (let i = 0; i <= totalEpochs; i++) {
          const progress = i / totalEpochs;
          history.push({
            epoch: i,
            accuracy: Math.min(0.96, 0.3 + progress * 0.6 + Math.sin(i * 0.2) * 0.05),
            loss: Math.max(0.02, 2.0 * Math.exp(-progress * 3) + Math.random() * 0.1),
            weights: generateRandomWeights(progress),
            gradients: Math.max(0.001, 1.0 * Math.exp(-progress * 2))
          });
        }
        break;

      case 'Logistic Regression':
        // Simulate logistic regression convergence
        for (let i = 0; i <= totalEpochs; i++) {
          const progress = i / totalEpochs;
          history.push({
            epoch: i,
            accuracy: Math.min(0.89, 0.4 + progress * 0.45 + Math.random() * 0.02),
            loss: Math.max(0.15, 1.5 * Math.exp(-progress * 2.5) + Math.random() * 0.05),
            weights: generateRandomWeights(progress),
            converged: progress > 0.8
          });
        }
        break;

      case 'K-Nearest Neighbor':
      case 'KNN':
        // KNN doesn't really "train", but simulate optimization
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.88, 0.5 + (i / totalEpochs) * 0.3 + Math.random() * 0.02),
            loss: Math.max(0.12, 0.8 - (i / totalEpochs) * 0.6 + Math.random() * 0.05),
            neighbors: Math.min(15, Math.floor(i / 3) + 1),
            distance: 'euclidean'
          });
        }
        break;

      case 'Random Forest':
        // Simulate ensemble building
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.93, 0.6 + (i / totalEpochs) * 0.3 + Math.sin(i * 0.15) * 0.01),
            loss: Math.max(0.07, 1.0 - (i / totalEpochs) * 0.85 + Math.random() * 0.05),
            trees: Math.min(100, i * 2 + 1),
            avgDepth: Math.min(12, Math.floor(i / 4) + 2)
          });
        }
        break;

      case 'Naive Bayes':
        // Simulate probability estimation
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.85, 0.5 + (i / totalEpochs) * 0.3 + Math.random() * 0.02),
            loss: Math.max(0.18, 1.2 - (i / totalEpochs) * 0.7 + Math.random() * 0.03),
            features: Math.min(50, i + 5),
            probabilities: 'updated'
          });
        }
        break;

      case 'Gradient Boosting':
        // Simulate boosting iterations
        for (let i = 0; i <= totalEpochs; i++) {
          const progress = i / totalEpochs;
          history.push({
            epoch: i,
            accuracy: Math.min(0.97, 0.3 + progress * 0.6 + Math.random() * 0.01),
            loss: Math.max(0.03, 1.5 * Math.exp(-progress * 2.8) + Math.random() * 0.02),
            estimators: i + 1,
            learningRate: 0.1
          });
        }
        break;

      default:
        // Default generic training curve
        for (let i = 0; i <= totalEpochs; i++) {
          history.push({
            epoch: i,
            accuracy: Math.min(0.9, 0.5 + (i / totalEpochs) * 0.4 + Math.random() * 0.03),
            loss: Math.max(0.1, 1.0 - (i / totalEpochs) * 0.8 + Math.random() * 0.05)
          });
        }
    }

    return history;
  };

  const generateRandomWeights = (progress) => {
    return Array.from({length: 10}, () =>
      (Math.random() - 0.5) * 2 * (1 - progress * 0.5)
    );
  };

  useEffect(() => {
    const history = simulateTraining(algorithm);
    setTrainingHistory(history);
    setCurrentEpoch(0);
  }, [algorithm]); // simulateTraining'i dependency'den çıkardık çünkü component içinde tanımlı


  const startAnimation = () => {
    if (isPlaying) return;

    setIsPlaying(true);
    intervalRef.current = setInterval(() => {
      setCurrentEpoch(prev => {
        if (prev >= trainingHistory.length - 1) {
          setIsPlaying(false);
          onComplete?.();
          return prev;
        }
        return prev + 1;
      });
    }, speed);
  };

  const stopAnimation = () => {
    setIsPlaying(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  const resetAnimation = () => {
    stopAnimation();
    setCurrentEpoch(0);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const currentData = trainingHistory.slice(0, currentEpoch + 1);

  const chartData = {
    labels: currentData.map(d => d.epoch),
    datasets: [
      {
        label: 'Accuracy',
        data: currentData.map(d => d.accuracy),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        yAxisID: 'y',
        tension: 0.1
      },
      {
        label: 'Loss',
        data: currentData.map(d => d.loss),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        yAxisID: 'y1',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: `${algorithm} Training Progress - Epoch ${currentEpoch}`,
        color: '#374151',
        font: { size: 14, weight: 'bold' }
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: { display: true, text: 'Accuracy', color: '#374151' },
        min: 0,
        max: 1,
        ticks: { color: '#6b7280' },
        grid: { color: 'rgba(107, 114, 128, 0.1)' }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: { display: true, text: 'Loss', color: '#374151' },
        min: 0,
        grid: { drawOnChartArea: false },
        ticks: { color: '#6b7280' }
      },
      x: {
        title: { display: true, text: 'Epoch', color: '#374151' },
        ticks: { color: '#6b7280' },
        grid: { color: 'rgba(107, 114, 128, 0.1)' }
      }
    },
    animation: {
      duration: 0 // Disable built-in animation for real-time feel
    }
  };

  const getCurrentStats = () => {
    if (!trainingHistory[currentEpoch]) return null;

    const current = trainingHistory[currentEpoch];
    switch(algorithm) {
      case 'Decision Tree':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">Tree Depth:</span>
              <span className="value">{current.treeDepth}</span>
            </div>
            <div className="stat-item">
              <span className="label">Nodes:</span>
              <span className="value">{current.nodes}</span>
            </div>
          </div>
        );

      case 'SVM':
      case 'Support Vector Machine':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">Margin:</span>
              <span className="value">{current.margin?.toFixed(3)}</span>
            </div>
            <div className="stat-item">
              <span className="label">Support Vectors:</span>
              <span className="value">{current.supportVectors}</span>
            </div>
          </div>
        );

      case 'Artificial Neural Network':
      case 'Neural Network':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">Gradient Norm:</span>
              <span className="value">{current.gradients?.toFixed(4)}</span>
            </div>
            <div className="stat-item">
              <span className="label">Weights Updated:</span>
              <span className="value">{current.weights?.length || 0}</span>
            </div>
          </div>
        );

      case 'Random Forest':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">Trees:</span>
              <span className="value">{current.trees}</span>
            </div>
            <div className="stat-item">
              <span className="label">Avg Depth:</span>
              <span className="value">{current.avgDepth}</span>
            </div>
          </div>
        );

      case 'K-Nearest Neighbor':
      case 'KNN':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">K Neighbors:</span>
              <span className="value">{current.neighbors}</span>
            </div>
            <div className="stat-item">
              <span className="label">Distance:</span>
              <span className="value">{current.distance}</span>
            </div>
          </div>
        );

      case 'Gradient Boosting':
        return (
          <div className="algorithm-stats">
            <div className="stat-item">
              <span className="label">Estimators:</span>
              <span className="value">{current.estimators}</span>
            </div>
            <div className="stat-item">
              <span className="label">Learning Rate:</span>
              <span className="value">{current.learningRate}</span>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="training-animation-container">
      <div className="animation-header">
        <h3>🎬 Live Training Simulation: {algorithm}</h3>
        <p>Watch how the algorithm learns step by step on {dataset}!</p>
      </div>

      <div className="animation-controls">
        <button
          onClick={startAnimation}
          disabled={isPlaying || currentEpoch >= trainingHistory.length - 1}
          className="control-btn play-btn"
        >
          {isPlaying ? '⏸️ Pause' : '▶️ Play'}
        </button>

        <button
          onClick={stopAnimation}
          disabled={!isPlaying}
          className="control-btn stop-btn"
        >
          ⏹️ Stop
        </button>

        <button
          onClick={resetAnimation}
          className="control-btn reset-btn"
        >
          🔄 Reset
        </button>

        <div className="speed-control">
          <label>Speed:</label>
          <input
            type="range"
            min="100"
            max="2000"
            step="100"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            disabled={isPlaying}
          />
          <span>{speed}ms</span>
        </div>
      </div>

      <div className="training-metrics">
        <div className="current-stats">
          <div className="stat-item primary">
            <span className="label">Current Epoch:</span>
            <span className="value">{currentEpoch}</span>
          </div>

          {trainingHistory[currentEpoch] && (
            <>
              <div className="stat-item">
                <span className="label">Accuracy:</span>
                <span className="value accuracy">
                  {trainingHistory[currentEpoch].accuracy.toFixed(4)}
                </span>
              </div>

              <div className="stat-item">
                <span className="label">Loss:</span>
                <span className="value loss">
                  {trainingHistory[currentEpoch].loss.toFixed(4)}
                </span>
              </div>
            </>
          )}

          {getCurrentStats()}
        </div>
      </div>

      <div className="chart-container" style={{ height: '400px' }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${(currentEpoch / (trainingHistory.length - 1)) * 100}%`
          }}
        ></div>
      </div>

      <style jsx>{`
        .training-animation-container {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: white;
        }

        .animation-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .animation-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.5rem;
        }

        .animation-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 16px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .control-btn {
          padding: 8px 16px;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.3s ease;
          backdrop-filter: blur(10px);
        }

        .control-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
          transform: translateY(-2px);
        }

        .control-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .speed-control {
          display: flex;
          align-items: center;
          gap: 8px;
          color: white;
        }

        .speed-control input {
          width: 100px;
        }

        .training-metrics {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
          backdrop-filter: blur(10px);
        }

        .current-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 16px;
        }

        .stat-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 6px;
        }

        .stat-item.primary {
          background: rgba(255, 255, 255, 0.2);
          font-weight: 600;
        }

        .label {
          font-size: 0.875rem;
          opacity: 0.9;
        }

        .value {
          font-weight: 600;
          font-size: 1rem;
        }

        .value.accuracy {
          color: #22c55e;
        }

        .value.loss {
          color: #ef4444;
        }

        .algorithm-stats {
          display: contents;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #22c55e, #3b82f6);
          transition: width 0.3s ease;
          border-radius: 4px;
        }

        @media (max-width: 768px) {
          .current-stats {
            grid-template-columns: 1fr;
          }

          .animation-controls {
            flex-direction: column;
            gap: 12px;
          }
        }
      `}</style>
    </div>
  );
};

export default TrainingAnimation;

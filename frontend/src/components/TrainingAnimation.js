import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

const TrainingAnimation = ({ algorithm, dataset, resultsData = [], onComplete }) => {
  const [epoch, setEpoch] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [accuracy, setAccuracy] = useState(0.1);
  const [loss, setLoss] = useState(2.0);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchProgress, setBatchProgress] = useState(0);
  const [trainingSpeed, setTrainingSpeed] = useState(1000);
  const [showMetrics, setShowMetrics] = useState(true);

  const [accuracyHistory, setAccuracyHistory] = useState([]);
  const [lossHistory, setLossHistory] = useState([]);
  const [lrHistory, setLrHistory] = useState([]);

  // Extract real training performance information
  const getTrainingPerformance = () => {
    if (resultsData.length === 0) return null;

    const algorithmResults = resultsData.filter(r => r.modelName === algorithm);
    if (algorithmResults.length === 0) return null;

    const bestResult = algorithmResults.reduce((best, current) => {
      const currentAcc = current.metrics?.accuracy || current.metrics?.Accuracy || 0;
      const bestAcc = best?.metrics?.accuracy || best?.metrics?.Accuracy || 0;
      return currentAcc >= bestAcc ? current : best;
    }, algorithmResults[0]);

    const accuracies = algorithmResults.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0).filter(a => a > 0);
    const trainingTimes = algorithmResults.map(r => r.fit_time_seconds || 0).filter(t => t > 0);
    const memoryUsages = algorithmResults.map(r => r.memory_usage_mb || 0).filter(m => m > 0);
    const throughputs = algorithmResults.map(r => r.training_throughput_samples_per_sec || 0).filter(t => t > 0);

    return {
      targetAccuracy: parseFloat(bestResult?.metrics?.accuracy || bestResult?.metrics?.Accuracy || 0.85),
      targetPrecision: parseFloat(bestResult?.metrics?.precision || bestResult?.metrics?.Precision || 0.8),
      targetRecall: parseFloat(bestResult?.metrics?.recall || bestResult?.metrics?.Recall || 0.8),
      targetF1Score: parseFloat(bestResult?.metrics?.f1_score || bestResult?.metrics?.['F1-Score'] || 0.8),
      avgAccuracy: accuracies.length > 0 ? accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length : 0.8,
      avgTrainingTime: trainingTimes.length > 0 ? trainingTimes.reduce((sum, t) => sum + t, 0) / trainingTimes.length : 5.0,
      avgMemoryUsage: memoryUsages.length > 0 ? memoryUsages.reduce((sum, m) => sum + m, 0) / memoryUsages.length : 100,
      avgThroughput: throughputs.length > 0 ? throughputs.reduce((sum, t) => sum + t, 0) / throughputs.length : 1000,
      totalInstances: algorithmResults.length,
      fromCache: bestResult?.from_cache || false,
      trainingMetrics: bestResult?.training_metrics || {},
      configId: bestResult?.configId || 'unknown'
    };
  };

  const performance = getTrainingPerformance();

  // Generate realistic training curves based on real data
  const generateTrainingCurve = () => {
    const maxEpochs = 100;
    const targetAcc = performance?.targetAccuracy || 0.85;
    const startingAcc = Math.max(0.1, targetAcc - 0.7);
    const convergenceRate = performance ? Math.log(targetAcc / startingAcc) / (maxEpochs * 0.7) : 0.05;

    const newAccuracyHistory = [];
    const newLossHistory = [];
    const newLrHistory = [];

    for (let i = 0; i <= maxEpochs; i++) {
      // Realistic accuracy curve with some noise
      let acc = startingAcc + (targetAcc - startingAcc) * (1 - Math.exp(-convergenceRate * i));
      acc += (Math.random() - 0.5) * 0.02; // Add noise
      acc = Math.min(Math.max(acc, 0), 1); // Clamp to [0, 1]

      // Corresponding loss curve
      let currentLoss = 2.5 * Math.exp(-i * 0.03) + Math.random() * 0.1;
      currentLoss = Math.max(currentLoss, 0.01);

      // Learning rate schedule (with decay)
      let currentLr = learningRate * Math.pow(0.95, Math.floor(i / 10));

      newAccuracyHistory.push(acc);
      newLossHistory.push(currentLoss);
      newLrHistory.push(currentLr);
    }

    return { newAccuracyHistory, newLossHistory, newLrHistory };
  };

  const startTraining = () => {
    if (isTraining) return;

    setIsTraining(true);
    setEpoch(0);
    setAccuracy(0.1);
    setLoss(2.0);
    setBatchProgress(0);

    const { newAccuracyHistory, newLossHistory, newLrHistory } = generateTrainingCurve();
    setAccuracyHistory([]);
    setLossHistory([]);
    setLrHistory([]);

    const maxEpochs = 100;
    let currentEpoch = 0;

    const interval = setInterval(() => {
      if (currentEpoch >= maxEpochs) {
        setIsTraining(false);
        clearInterval(interval);
        if (onComplete) onComplete();
        return;
      }

      // Update current metrics
      setEpoch(currentEpoch);
      setAccuracy(newAccuracyHistory[currentEpoch]);
      setLoss(newLossHistory[currentEpoch]);
      setLearningRate(newLrHistory[currentEpoch]);

      // Update history
      setAccuracyHistory(prev => [...prev, newAccuracyHistory[currentEpoch]]);
      setLossHistory(prev => [...prev, newLossHistory[currentEpoch]]);
      setLrHistory(prev => [...prev, newLrHistory[currentEpoch]]);

      // Simulate batch progress
      setBatchProgress(0);
      const batchInterval = setInterval(() => {
        setBatchProgress(prev => {
          if (prev >= 100) {
            clearInterval(batchInterval);
            return 100;
          }
          return prev + 10;
        });
      }, trainingSpeed / 10);

      currentEpoch++;
    }, trainingSpeed);
  };

  const chartData = {
    labels: Array.from({ length: accuracyHistory.length }, (_, i) => i),
    datasets: [
      {
        label: 'Training Accuracy',
        data: accuracyHistory,
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        pointRadius: 1,
        pointHoverRadius: 4,
        borderWidth: 2
      },
      {
        label: 'Validation Loss',
        data: lossHistory.map(loss => loss / 3), // Scale for visualization
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        pointRadius: 1,
        pointHoverRadius: 4,
        borderWidth: 2,
        yAxisID: 'y1'
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: `${algorithm} Training Progress on ${dataset}`,
        font: {
          size: 14,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            if (label === 'Training Accuracy') {
              return `${label}: ${(context.parsed.y * 100).toFixed(2)}%`;
            } else {
              return `${label}: ${(context.parsed.y * 3).toFixed(4)}`;
            }
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Epoch'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Accuracy'
        },
        min: 0,
        max: 1,
        grid: {
          color: 'rgba(34, 197, 94, 0.2)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Loss (scaled)'
        },
        min: 0,
        max: 1,
        grid: {
          drawOnChartArea: false,
          color: 'rgba(239, 68, 68, 0.2)'
        }
      }
    }
  };

  return (
    <div className="training-animation-container">
      <div className="training-header">
        <h3>🔥 Training Progress</h3>
        <p>
          Real-time simulation of <strong>{algorithm}</strong> learning <strong>{dataset}</strong>
          {performance && ` (Target Accuracy: ${(performance.targetAccuracy * 100).toFixed(1)}%)`}
        </p>
      </div>

      {/* Real Performance Metrics */}
      {performance && showMetrics && (
        <div className="real-performance-panel">
          <div className="panel-header">
            <h4>🎯 Actual Experiment Data</h4>
            <button
              className="toggle-panel-btn"
              onClick={() => setShowMetrics(!showMetrics)}
            >
              {showMetrics ? '🙈 Hide' : '👁️ Show'}
            </button>
          </div>
          <div className="performance-metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Target Accuracy</div>
              <div className="metric-value" style={{color: getMetricColor(performance.targetAccuracy)}}>
                {(performance.targetAccuracy * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Target Precision</div>
              <div className="metric-value" style={{color: getMetricColor(performance.targetPrecision)}}>
                {(performance.targetPrecision * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Target Recall</div>
              <div className="metric-value" style={{color: getMetricColor(performance.targetRecall)}}>
                {(performance.targetRecall * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">F1-Score</div>
              <div className="metric-value" style={{color: getMetricColor(performance.targetF1Score)}}>
                {(performance.targetF1Score * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Training Time</div>
              <div className="metric-value">{performance.avgTrainingTime.toFixed(2)}s</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Memory Usage</div>
              <div className="metric-value">{performance.avgMemoryUsage.toFixed(1)} MB</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Throughput</div>
              <div className="metric-value">{performance.avgThroughput.toFixed(0)} samples/s</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Cache Status</div>
              <div className="metric-value cache-status">
                {performance.fromCache ? '🚀 CACHED' : '💻 COMPUTED'}
              </div>
            </div>
          </div>
          <div className="experiment-summary">
            <strong>Experiment Summary:</strong> {performance.totalInstances} instance{performance.totalInstances !== 1 ? 's' : ''} of {algorithm} tested
            | Config ID: {performance.configId}
          </div>
        </div>
      )}

      <div className="training-controls">
        <button
          onClick={startTraining}
          disabled={isTraining}
          className="start-training-btn"
        >
          {isTraining ? '🔄 Training...' : '⚡ Start Training'}
        </button>

        <div className="speed-controls">
          <label>Animation Speed:</label>
          <select
            value={trainingSpeed}
            onChange={(e) => setTrainingSpeed(Number(e.target.value))}
            disabled={isTraining}
          >
            <option value={2000}>Slow (2s/epoch)</option>
            <option value={1000}>Normal (1s/epoch)</option>
            <option value={500}>Fast (0.5s/epoch)</option>
            <option value={200}>Very Fast (0.2s/epoch)</option>
          </select>
        </div>

        <div className="current-metrics">
          <span>Epoch: <strong>{epoch}/100</strong></span>
          <span>Accuracy: <strong>{(accuracy * 100).toFixed(2)}%</strong></span>
          <span>Loss: <strong>{loss.toFixed(4)}</strong></span>
          <span>LR: <strong>{learningRate.toExponential(2)}</strong></span>
        </div>
      </div>

      <div className="training-progress">
        <div className="progress-section">
          <div className="progress-label">Batch Progress</div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${batchProgress}%` }}
            ></div>
          </div>
          <div className="progress-text">{batchProgress}%</div>
        </div>

        <div className="progress-section">
          <div className="progress-label">Epoch Progress</div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(epoch / 100) * 100}%` }}
            ></div>
          </div>
          <div className="progress-text">{epoch}/100</div>
        </div>
      </div>

      <div className="chart-container" style={{ height: '350px' }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="training-insights">
        <h4>📈 Training Insights</h4>
        <div className="insights-grid">
          <div className="insight-item">
            <strong>Algorithm Type:</strong> {algorithm}
            <small>
              {algorithm.toLowerCase().includes('tree') && 'Ensemble method with high interpretability'}
              {algorithm.toLowerCase().includes('svm') && 'Support Vector Machine with kernel tricks'}
              {algorithm.toLowerCase().includes('neural') && 'Multi-layer perceptron with backpropagation'}
              {algorithm.toLowerCase().includes('linear') && 'Linear model with fast convergence'}
              {algorithm.toLowerCase().includes('random') && 'Ensemble of randomized decision trees'}
              {!['tree', 'svm', 'neural', 'linear', 'random'].some(type => algorithm.toLowerCase().includes(type)) && 'Advanced machine learning algorithm'}
            </small>
          </div>
          <div className="insight-item">
            <strong>Convergence Pattern:</strong>
            {performance && performance.targetAccuracy > 0.9 ? 'Fast Convergence' :
             performance && performance.targetAccuracy > 0.8 ? 'Steady Convergence' : 'Gradual Learning'}
            <small>
              {performance && performance.targetAccuracy > 0.9 && 'Model achieves high accuracy quickly'}
              {performance && performance.targetAccuracy <= 0.9 && performance.targetAccuracy > 0.8 && 'Consistent improvement throughout training'}
              {performance && performance.targetAccuracy <= 0.8 && 'Model needs more training or tuning'}
            </small>
          </div>
          <div className="insight-item">
            <strong>Performance Level:</strong>
            {performance && performance.targetAccuracy > 0.95 ? 'Excellent' :
             performance && performance.targetAccuracy > 0.9 ? 'Very Good' :
             performance && performance.targetAccuracy > 0.8 ? 'Good' : 'Needs Improvement'}
            <small>
              Based on actual experimental results from your simulation
            </small>
          </div>
        </div>
      </div>

      <style jsx>{`
        .training-animation-container {
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: white;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .training-header {
          text-align: center;
          margin-bottom: 24px;
        }

        .training-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.4rem;
          font-weight: 700;
        }

        .training-header p {
          margin: 0;
          font-size: 1rem;
          opacity: 0.9;
          line-height: 1.5;
        }

        .real-performance-panel {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .panel-header h4 {
          margin: 0;
          font-size: 1.1rem;
        }

        .toggle-panel-btn {
          padding: 4px 8px;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .performance-metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 12px;
          text-align: center;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .metric-label {
          font-size: 0.8rem;
          opacity: 0.9;
          margin-bottom: 6px;
        }

        .metric-value {
          font-size: 1rem;
          font-weight: 700;
          font-family: monospace;
        }

        .cache-status {
          color: #22c55e !important;
        }

        .experiment-summary {
          text-align: center;
          font-size: 0.9rem;
          opacity: 0.9;
          background: rgba(255, 255, 255, 0.05);
          padding: 8px 12px;
          border-radius: 6px;
        }

        .training-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
          margin-bottom: 24px;
          flex-wrap: wrap;
          background: rgba(255, 255, 255, 0.1);
          padding: 16px;
          border-radius: 8px;
          backdrop-filter: blur(5px);
        }

        .start-training-btn {
          padding: 12px 24px;
          border: none;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.3s ease;
          border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .start-training-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .start-training-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
        }

        .speed-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .speed-controls label {
          font-size: 0.9rem;
          font-weight: 600;
        }

        .speed-controls select {
          padding: 4px 8px;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 4px;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 0.9rem;
        }

        .current-metrics {
          display: flex;
          gap: 16px;
          font-size: 0.9rem;
          font-weight: 500;
          flex-wrap: wrap;
        }

        .training-progress {
          display: flex;
          gap: 24px;
          margin-bottom: 24px;
          flex-wrap: wrap;
        }

        .progress-section {
          flex: 1;
          min-width: 200px;
        }

        .progress-label {
          font-size: 0.9rem;
          font-weight: 600;
          margin-bottom: 8px;
          opacity: 0.9;
        }

        .progress-bar {
          width: 100%;
          height: 20px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
          overflow: hidden;
          position: relative;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
          transition: width 0.3s ease;
          border-radius: 10px;
        }

        .progress-text {
          text-align: center;
          font-size: 0.8rem;
          font-weight: 600;
          margin-top: 4px;
          opacity: 0.9;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .training-insights {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 20px;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .training-insights h4 {
          margin: 0 0 16px 0;
          font-size: 1.1rem;
          text-align: center;
        }

        .insights-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 16px;
        }

        .insight-item {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
          border-left: 4px solid rgba(255, 255, 255, 0.5);
        }

        .insight-item strong {
          display: block;
          margin-bottom: 6px;
          font-size: 0.95rem;
        }

        .insight-item small {
          display: block;
          opacity: 0.8;
          font-size: 0.8rem;
          line-height: 1.4;
          margin-top: 4px;
        }

        @media (max-width: 768px) {
          .training-controls {
            flex-direction: column;
            gap: 12px;
          }

          .current-metrics {
            flex-direction: column;
            gap: 8px;
            text-align: center;
          }

          .training-progress {
            flex-direction: column;
            gap: 16px;
          }

          .performance-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
          }

          .insights-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

const getMetricColor = (value) => {
  if (value >= 0.9) return '#22c55e';
  if (value >= 0.8) return '#3b82f6';
  if (value >= 0.7) return '#f59e0b';
  return '#ef4444';
};

export default TrainingAnimation;

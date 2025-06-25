import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

const TrainingAnimation = ({ algorithm, dataset, resultsData = [], onComplete }) => {
  const [epoch, setEpoch] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [accuracy, setAccuracy] = useState(0.1);
  const [loss, setLoss] = useState(2.0);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchProgress, setBatchProgress] = useState(0);
  const [trainingSpeed, setTrainingSpeed] = useState(500);
  const [showMetrics, setShowMetrics] = useState(true);
  const [useRealData, setUseRealData] = useState(true);

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
      configId: bestResult?.configId || 'unknown',
      // NEW: Extract real epoch data
      epochData: bestResult?.epoch_data || null,
      learningCurve: bestResult?.learning_curve || null
    };
  };

  const performance = getTrainingPerformance();

const getRealEpochData = () => {
    if (!performance?.epochData && !performance?.learningCurve) return null;

    // First try to get epoch data directly from the result
    let epochData = null;

    // Check all results for epoch data
    for (const result of resultsData) {
      if (result.epoch_data && result.epoch_data.epochs && result.epoch_data.epochs.length > 0) {
        epochData = result.epoch_data;
        console.log(`Found real epoch data for ${algorithm}: ${epochData.epochs.length} epochs`);
        break;
      }
    }

    // If no epoch data found in results, check performance object
    if (!epochData && performance.epochData && performance.epochData.epochs) {
      epochData = performance.epochData;
      console.log(`Using epoch data from performance: ${epochData.epochs.length} epochs`);
    }

    // Validate epoch data structure
    if (!epochData || !epochData.epochs || !epochData.train_accuracy) {
      console.log("No valid epoch data found, will use synthetic data");
      return null;
    }

    // Convert to expected format
    const realEpochData = {
      epochs: epochData.epochs || [],
      trainAccuracy: epochData.train_accuracy || [],
      valAccuracy: epochData.val_accuracy || epochData.train_accuracy?.map(acc => acc * 0.95) || [],
      trainLoss: epochData.train_loss || epochData.train_accuracy?.map(acc => 1 - acc) || [],
      valLoss: epochData.val_loss || epochData.val_accuracy?.map(acc => 1 - acc) || [],
      learningRates: epochData.learning_rates || epochData.epochs?.map((_, i) => 0.001 * Math.pow(0.95, i)) || [],
      totalEpochs: epochData.total_epochs || epochData.epochs.length,
      finalTrainAcc: epochData.final_train_acc || (epochData.train_accuracy?.length > 0 ? epochData.train_accuracy[epochData.train_accuracy.length - 1] : 0.8),
      finalValAcc: epochData.final_val_acc || (epochData.val_accuracy?.length > 0 ? epochData.val_accuracy[epochData.val_accuracy.length - 1] : 0.8),
      bestValAcc: epochData.best_val_acc || Math.max(...(epochData.val_accuracy || [0.8])),
      bestEpoch: epochData.best_epoch || 0,
      isSynthetic: epochData.is_synthetic === true
    };

    console.log(`Real epoch data processed: ${realEpochData.totalEpochs} epochs, best val acc: ${realEpochData.bestValAcc.toFixed(3)}, synthetic: ${realEpochData.isSynthetic}`);

    return realEpochData;
  };

  const realEpochData = getRealEpochData();

const generateTrainingCurve = () => {
    if (useRealData && realEpochData && !realEpochData.isSynthetic) {
      console.log("Using REAL epoch data for training animation");
      return {
        newAccuracyHistory: realEpochData.trainAccuracy,
        newValAccuracyHistory: realEpochData.valAccuracy,
        newLossHistory: realEpochData.trainLoss,
        newValLossHistory: realEpochData.valLoss,
        newLrHistory: realEpochData.learningRates,
        totalEpochs: realEpochData.totalEpochs
      };
    } else {
      console.log("Using synthetic data for training animation");
      // Enhanced synthetic data generation based on actual performance
      const maxEpochs = realEpochData?.totalEpochs || 50;
      const targetAcc = performance?.targetAccuracy || 0.85;
      const startingAcc = Math.max(0.1, targetAcc - 0.6);

      // Algorithm-specific convergence patterns
      let convergenceRate = 0.1;
      let noiseLevel = 0.02;

      if (algorithm.includes('Neural') || algorithm.includes('ANN')) {
        convergenceRate = 0.08;
        noiseLevel = 0.03;
      } else if (algorithm.includes('Tree')) {
        convergenceRate = 0.15;
        noiseLevel = 0.015;
      } else if (algorithm.includes('SVM')) {
        convergenceRate = 0.12;
        noiseLevel = 0.01;
      }

      const newAccuracyHistory = [];
      const newValAccuracyHistory = [];
      const newLossHistory = [];
      const newValLossHistory = [];
      const newLrHistory = [];

      for (let i = 0; i <= maxEpochs; i++) {
        // Progressive improvement with algorithm-specific characteristics
        let acc = startingAcc + (targetAcc - startingAcc) * (1 - Math.exp(-convergenceRate * i));
        acc += (Math.random() - 0.5) * noiseLevel; // Add noise
        acc = Math.min(Math.max(acc, 0.1), 1); // Clamp to [0, 1]

        // Validation accuracy (with generalization gap)
        let valAcc = acc * (0.92 + Math.random() * 0.08);
        valAcc = Math.min(Math.max(valAcc, 0.1), 1);

        // Corresponding loss curves
        let currentLoss = Math.max(0.01, 2.5 * Math.exp(-i * 0.05) + Math.random() * 0.1);
        let valLoss = currentLoss * (1.1 + Math.random() * 0.1);

        // Learning rate schedule
        let currentLr = learningRate * Math.pow(0.96, Math.floor(i / 10));

        newAccuracyHistory.push(acc);
        newValAccuracyHistory.push(valAcc);
        newLossHistory.push(currentLoss);
        newValLossHistory.push(valLoss);
        newLrHistory.push(currentLr);
      }

      // Ensure final accuracy matches target
      if (newAccuracyHistory.length > 0) {
        newAccuracyHistory[newAccuracyHistory.length - 1] = targetAcc;
        newValAccuracyHistory[newValAccuracyHistory.length - 1] = targetAcc * 0.95;
      }

      return {
        newAccuracyHistory,
        newValAccuracyHistory,
        newLossHistory,
        newValLossHistory,
        newLrHistory,
        totalEpochs: maxEpochs
      };
    }
  };

  const startTraining = () => {
    if (isTraining) return;

    setIsTraining(true);
    setEpoch(0);
    setAccuracy(0.1);
    setLoss(2.0);
    setBatchProgress(0);

    const curveData = generateTrainingCurve();
    const {
      newAccuracyHistory,
      newValAccuracyHistory,
      newLossHistory,
      newValLossHistory,
      newLrHistory,
      totalEpochs
    } = curveData;

    setAccuracyHistory([]);
    setLossHistory([]);
    setLrHistory([]);

    const maxEpochs = totalEpochs;
    let currentEpoch = 0;

    const interval = setInterval(() => {
      if (currentEpoch >= maxEpochs) {
        setIsTraining(false);
        clearInterval(interval);
        if (onComplete) onComplete();
        return;
      }

      // Update current metrics using real or synthetic data
      setEpoch(currentEpoch);
      setAccuracy(newAccuracyHistory[currentEpoch] || 0.1);
      setLoss(newLossHistory[currentEpoch] || 2.0);
      setLearningRate(newLrHistory[currentEpoch] || 0.001);

      // Update history for charts
      setAccuracyHistory(prev => [...prev, newAccuracyHistory[currentEpoch] || 0.1]);
      setLossHistory(prev => [...prev, newLossHistory[currentEpoch] || 2.0]);
      setLrHistory(prev => [...prev, newLrHistory[currentEpoch] || 0.001]);

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
        label: 'Validation Accuracy',
        data: accuracyHistory.map((_, i) => {
          // Use validation data if available
          if (useRealData && realEpochData && realEpochData.valAccuracy[i]) {
            return realEpochData.valAccuracy[i];
          }
          return accuracyHistory[i] * 0.95; // Approximate validation accuracy
        }),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        pointRadius: 1,
        pointHoverRadius: 4,
        borderWidth: 2
      },
      {
        label: 'Training Loss',
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
        text: `${algorithm} Training Progress on ${dataset} ${realEpochData && !realEpochData.isSynthetic ? '(Real Data)' : '(Synthetic)'}`,
        font: {
          size: 14,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            if (label.includes('Accuracy')) {
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
          {realEpochData && !realEpochData.isSynthetic ? (
            <>
              <strong>Real-time simulation</strong> using actual epoch data from <strong>{algorithm}</strong> learning <strong>{dataset}</strong>
              {performance && ` (Target Accuracy: ${(performance.targetAccuracy * 100).toFixed(1)}%)`}
            </>
          ) : (
            <>
              <strong>Synthetic simulation</strong> of <strong>{algorithm}</strong> learning <strong>{dataset}</strong>
              {performance && ` (Target Accuracy: ${(performance.targetAccuracy * 100).toFixed(1)}%)`}
            </>
          )}
        </p>
        {realEpochData && !realEpochData.isSynthetic && (
          <div className="real-data-badge">
            ✅ Using Real Training Data ({realEpochData.totalEpochs} epochs recorded)
          </div>
        )}
      </div>

      {/* Real Performance Metrics */}
      {performance && showMetrics && (
        <div className="real-performance-panel">
          <div className="panel-header">
            <h4>🎯 Actual Experiment Data</h4>
            <div className="panel-controls">
              <button
                className="toggle-panel-btn"
                onClick={() => setShowMetrics(!showMetrics)}
              >
                {showMetrics ? '🙈 Hide' : '👁️ Show'}
              </button>
              {realEpochData && (
                <button
                  className="toggle-data-type-btn"
                  onClick={() => setUseRealData(!useRealData)}
                  disabled={isTraining}
                >
                  {useRealData ? '📊 Real Data' : '🎨 Synthetic'}
                </button>
              )}
            </div>
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
            {realEpochData && !realEpochData.isSynthetic && (
              <span> | <strong>Real Epoch Data:</strong> {realEpochData.totalEpochs} epochs, Best Val Acc: {(realEpochData.bestValAcc * 100).toFixed(1)}%</span>
            )}
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
          <span>Epoch: <strong>{epoch}/{realEpochData?.totalEpochs || 100}</strong></span>
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
              style={{ width: `${(epoch / (realEpochData?.totalEpochs || 100)) * 100}%` }}
            ></div>
          </div>
          <div className="progress-text">{epoch}/{realEpochData?.totalEpochs || 100}</div>
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
            <strong>Data Source:</strong>
            {realEpochData && !realEpochData.isSynthetic ? 'Real Epoch Data' : 'Synthetic Simulation'}
            <small>
              {realEpochData && !realEpochData.isSynthetic
                ? `Animation uses actual training data collected during model training with ${realEpochData.totalEpochs} epochs`
                : 'Animation uses synthetic data based on final model performance metrics'
              }
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
              Based on {realEpochData && !realEpochData.isSynthetic ? 'real training data' : 'experimental results'} from your simulation
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

        .real-data-badge {
          margin-top: 12px;
          padding: 8px 16px;
          background: rgba(34, 197, 94, 0.2);
          border: 2px solid rgba(34, 197, 94, 0.5);
          border-radius: 8px;
          font-size: 0.9rem;
          font-weight: 600;
          display: inline-block;
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

        .panel-controls {
          display: flex;
          gap: 8px;
        }

        .toggle-panel-btn, .toggle-data-type-btn {
          padding: 4px 8px;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-size: 0.8rem;
          font-weight: 600;
          transition: background-color 0.3s ease;
        }

        .toggle-data-type-btn {
          background: rgba(34, 197, 94, 0.2);
          border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .toggle-panel-btn:hover, .toggle-data-type-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
        }

        .toggle-data-type-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
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

          .panel-controls {
            flex-direction: column;
            gap: 4px;
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

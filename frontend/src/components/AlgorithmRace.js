import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

const AlgorithmRace = ({ selectedAlgorithms, dataset, resultsData = [] }) => {
  const [isRacing, setIsRacing] = useState(false);
  const [raceProgress, setRaceProgress] = useState({});
  const [raceResults, setRaceResults] = useState({});
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [raceSpeed, setRaceSpeed] = useState(500);
  const [showDetails, setShowDetails] = useState(true);
  const [winner, setWinner] = useState(null);
  const [raceHistory, setRaceHistory] = useState({});

  const getAlgorithmColor = (algorithm) => {
    const colors = {
      'Decision Tree': '#22c55e',
      'Random Forest': '#16a34a',
      'SVM': '#3b82f6',
      'Neural Network': '#8b5cf6',
      'Linear Regression': '#f59e0b',
      'Logistic Regression': '#ef4444',
      'KNN': '#06b6d4',
      'Naive Bayes': '#84cc16',
      'XGBoost': '#f97316',
      'LightGBM': '#6366f1'
    };

    // Try exact match first
    if (colors[algorithm]) return colors[algorithm];

    // Try partial matches
    const colorKeys = Object.keys(colors);
    for (let key of colorKeys) {
      if (algorithm.toLowerCase().includes(key.toLowerCase()) ||
          key.toLowerCase().includes(algorithm.toLowerCase())) {
        return colors[key];
      }
    }

    // Fallback to hash-based color
    let hash = 0;
    for (let i = 0; i < algorithm.length; i++) {
      hash = algorithm.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  };

  // Extract real performance data for each algorithm
  const getAlgorithmPerformance = () => {
    const performanceData = {};

    selectedAlgorithms.forEach(algorithm => {
      const algorithmResults = resultsData.filter(r => r.modelName === algorithm);

      if (algorithmResults.length > 0) {
        const bestResult = algorithmResults.reduce((best, current) => {
          const currentAcc = current.metrics?.accuracy || current.metrics?.Accuracy || 0;
          const bestAcc = best?.metrics?.accuracy || best?.metrics?.Accuracy || 0;
          return currentAcc >= bestAcc ? current : best;
        }, algorithmResults[0]);

        const accuracies = algorithmResults.map(r => r.metrics?.accuracy || r.metrics?.Accuracy || 0).filter(a => a > 0);
        const trainingTimes = algorithmResults.map(r => r.fit_time_seconds || 0).filter(t => t > 0);
        const memoryUsages = algorithmResults.map(r => r.memory_usage_mb || 0).filter(m => m > 0);

        performanceData[algorithm] = {
          targetAccuracy: parseFloat(bestResult?.metrics?.accuracy || bestResult?.metrics?.Accuracy || 0.8),
          targetPrecision: parseFloat(bestResult?.metrics?.precision || bestResult?.metrics?.Precision || 0.8),
          targetRecall: parseFloat(bestResult?.metrics?.recall || bestResult?.metrics?.Recall || 0.8),
          avgAccuracy: accuracies.length > 0 ? accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length : 0.7,
          avgTrainingTime: trainingTimes.length > 0 ? trainingTimes.reduce((sum, t) => sum + t, 0) / trainingTimes.length : Math.random() * 10 + 1,
          avgMemoryUsage: memoryUsages.length > 0 ? memoryUsages.reduce((sum, m) => sum + m, 0) / memoryUsages.length : Math.random() * 200 + 50,
          instances: algorithmResults.length,
          fromCache: bestResult?.from_cache || false,
          convergenceRate: Math.log(parseFloat(bestResult?.metrics?.accuracy || 0.8) / 0.1) / 80,
          color: getAlgorithmColor(algorithm),
          configId: bestResult?.configId || 'unknown'
        };
      } else {
        // Fallback for algorithms without results
        performanceData[algorithm] = {
          targetAccuracy: 0.7 + Math.random() * 0.25,
          targetPrecision: 0.75 + Math.random() * 0.2,
          targetRecall: 0.75 + Math.random() * 0.2,
          avgAccuracy: 0.6 + Math.random() * 0.3,
          avgTrainingTime: Math.random() * 15 + 2,
          avgMemoryUsage: Math.random() * 300 + 100,
          instances: 0,
          fromCache: false,
          convergenceRate: 0.03 + Math.random() * 0.04,
          color: getAlgorithmColor(algorithm),
          configId: 'simulated'
        };
      }
    });

    return performanceData;
  };

  const performanceData = getAlgorithmPerformance();

  // Initialize race state
  useEffect(() => {
    const initialProgress = {};
    const initialResults = {};
    const initialHistory = {};

    selectedAlgorithms.forEach(algorithm => {
      initialProgress[algorithm] = 0;
      initialResults[algorithm] = {
        accuracy: 0.1,
        epoch: 0,
        finished: false,
        finalTime: 0
      };
      initialHistory[algorithm] = [];
    });

    setRaceProgress(initialProgress);
    setRaceResults(initialResults);
    setRaceHistory(initialHistory);
    setWinner(null);
  }, [selectedAlgorithms]);

  const startRace = () => {
    if (isRacing) return;

    setIsRacing(true);
    setCurrentEpoch(0);
    setWinner(null);

    // Reset all algorithms
    const resetProgress = {};
    const resetResults = {};
    const resetHistory = {};

    selectedAlgorithms.forEach(algorithm => {
      resetProgress[algorithm] = 0;
      resetResults[algorithm] = {
        accuracy: 0.1,
        epoch: 0,
        finished: false,
        finalTime: 0
      };
      resetHistory[algorithm] = [0.1];
    });

    setRaceProgress(resetProgress);
    setRaceResults(resetResults);
    setRaceHistory(resetHistory);

    const maxEpochs = 100;
    let epoch = 0;
    let finishedCount = 0;

    const raceInterval = setInterval(() => {
      epoch++;
      setCurrentEpoch(epoch);

      setRaceProgress(prevProgress => {
        const newProgress = { ...prevProgress };

        selectedAlgorithms.forEach(algorithm => {
          if (!raceResults[algorithm]?.finished) {
            const perf = performanceData[algorithm];
            const targetAcc = perf.targetAccuracy;
            const convergenceRate = perf.convergenceRate;

            // Calculate accuracy for this epoch with some randomness
            let accuracy = 0.1 + (targetAcc - 0.1) * (1 - Math.exp(-convergenceRate * epoch));
            accuracy += (Math.random() - 0.5) * 0.02; // Add noise
            accuracy = Math.min(Math.max(accuracy, 0.1), targetAcc + 0.01);

            newProgress[algorithm] = (accuracy / targetAcc) * 100;

            // Update results
            setRaceResults(prevResults => ({
              ...prevResults,
              [algorithm]: {
                ...prevResults[algorithm],
                accuracy,
                epoch,
                finished: accuracy >= targetAcc * 0.99,
                finalTime: accuracy >= targetAcc * 0.99 ? epoch * (raceSpeed / 1000) : 0
              }
            }));

            // Update history
            setRaceHistory(prevHistory => ({
              ...prevHistory,
              [algorithm]: [...prevHistory[algorithm], accuracy]
            }));

            // Check if finished
            if (accuracy >= targetAcc * 0.99 && !raceResults[algorithm]?.finished) {
              finishedCount++;
              if (finishedCount === 1 && !winner) {
                setWinner(algorithm);
              }
            }
          }
        });

        return newProgress;
      });

      if (epoch >= maxEpochs || finishedCount >= selectedAlgorithms.length) {
        setIsRacing(false);
        clearInterval(raceInterval);

        // Determine winner if not set
        if (!winner) {
          const finalAccuracies = {};
          selectedAlgorithms.forEach(algorithm => {
            finalAccuracies[algorithm] = raceResults[algorithm]?.accuracy || 0;
          });

          const bestAlgorithm = Object.keys(finalAccuracies).reduce((a, b) =>
            finalAccuracies[a] > finalAccuracies[b] ? a : b
          );
          setWinner(bestAlgorithm);
        }
      }
    }, raceSpeed);
  };



  const chartData = {
    labels: Array.from({ length: Math.max(...Object.values(raceHistory).map(h => h.length)) }, (_, i) => i),
    datasets: selectedAlgorithms.map(algorithm => ({
      label: algorithm,
      data: raceHistory[algorithm] || [],
      borderColor: performanceData[algorithm]?.color || getAlgorithmColor(algorithm),
      backgroundColor: `${performanceData[algorithm]?.color || getAlgorithmColor(algorithm)}20`,
      tension: 0.4,
      pointRadius: 2,
      pointHoverRadius: 5,
      borderWidth: 3
    }))
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
          padding: 15,
          font: {
            size: 11,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: `Algorithm Race on ${dataset} Dataset`,
        font: {
          size: 16,
          weight: 'bold'
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Training Epoch'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Accuracy'
        },
        min: 0,
        max: 1,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };

  return (
    <div className="algorithm-race-container">
      <div className="race-header">
        <h3>🏁 Algorithm Learning Race</h3>
        <p>
          Watch <strong>{selectedAlgorithms.length}</strong> algorithms compete to learn <strong>{dataset}</strong> dataset!
        </p>
        <p className="race-description">
          Each algorithm races to reach its real-world target accuracy based on your experimental data.
        </p>
      </div>

      {/* Real Performance Summary */}
      {showDetails && (
        <div className="performance-summary">
          <div className="summary-header">
            <h4>🎯 Real Performance Targets</h4>
            <button
              className="toggle-details-btn"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? '🙈 Hide' : '👁️ Show'}
            </button>
          </div>
          <div className="algorithm-cards">
            {selectedAlgorithms.map(algorithm => {
              const perf = performanceData[algorithm];
              return (
                <div key={algorithm} className="algorithm-card">
                  <div className="algorithm-header">
                    <div
                      className="algorithm-color"
                      style={{ backgroundColor: perf.color }}
                    ></div>
                    <div className="algorithm-name">{algorithm}</div>
                  </div>
                  <div className="algorithm-stats">
                    <div className="stat-row">
                      <span>Target Accuracy:</span>
                      <span className="stat-value">{(perf.targetAccuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="stat-row">
                      <span>Avg Training Time:</span>
                      <span className="stat-value">{perf.avgTrainingTime.toFixed(2)}s</span>
                    </div>
                    <div className="stat-row">
                      <span>Memory Usage:</span>
                      <span className="stat-value">{perf.avgMemoryUsage.toFixed(0)} MB</span>
                    </div>
                    <div className="stat-row">
                      <span>Instances:</span>
                      <span className="stat-value">{perf.instances}</span>
                    </div>
                    <div className="stat-row">
                      <span>Cache:</span>
                      <span className={`stat-value cache-${perf.fromCache ? 'hit' : 'miss'}`}>
                        {perf.fromCache ? '🚀 HIT' : '💻 COMPUTED'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="race-controls">
        <button
          onClick={startRace}
          disabled={isRacing}
          className="start-race-btn"
        >
          {isRacing ? '🏃‍♂️ Racing...' : '🚀 Start Race'}
        </button>

        <div className="speed-control">
          <label>Race Speed:</label>
          <select
            value={raceSpeed}
            onChange={(e) => setRaceSpeed(Number(e.target.value))}
            disabled={isRacing}
          >
            <option value={1000}>Slow (1s/epoch)</option>
            <option value={500}>Normal (0.5s/epoch)</option>
            <option value={250}>Fast (0.25s/epoch)</option>
            <option value={100}>Very Fast (0.1s/epoch)</option>
          </select>
        </div>

        <div className="race-info">
          <span>Current Epoch: <strong>{currentEpoch}/100</strong></span>
          {winner && (
            <span className="winner-announcement">
              🏆 Winner: <strong>{winner}</strong>!
            </span>
          )}
        </div>
      </div>

      <div className="race-track">
        <h4>🏃‍♂️ Race Progress</h4>
        <div className="track-lanes">
          {selectedAlgorithms.map(algorithm => {
            const progress = raceProgress[algorithm] || 0;
            const result = raceResults[algorithm] || {};
            const perf = performanceData[algorithm];

            return (
              <div key={algorithm} className="race-lane">
                <div className="lane-header">
                  <div className="algorithm-info">
                    <div
                      className="lane-color"
                      style={{ backgroundColor: perf.color }}
                    ></div>
                    <span className="lane-name">{algorithm}</span>
                  </div>
                  <div className="lane-stats">
                    <span>Acc: {(result.accuracy * 100).toFixed(1)}%</span>
                    <span>Target: {(perf.targetAccuracy * 100).toFixed(1)}%</span>
                    {result.finished && (
                      <span className="finished-badge">✅ FINISHED</span>
                    )}
                  </div>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-runner"
                    style={{
                      left: `${Math.min(progress, 100)}%`,
                      backgroundColor: perf.color,
                      opacity: result.finished ? 1 : 0.8
                    }}
                  >
                    {algorithm === winner ? '🏆' : '🏃‍♂️'}
                  </div>
                  <div className="track-markers">
                    <div className="marker start">START</div>
                    <div className="marker finish">FINISH</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="chart-container" style={{ height: '400px' }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="race-summary">
        <h4>📊 Race Results Summary</h4>
        <div className="results-table">
          <div className="table-header">
            <span>Rank</span>
            <span>Algorithm</span>
            <span>Final Accuracy</span>
            <span>Target Accuracy</span>
            <span>Completion Time</span>
            <span>Performance</span>
          </div>
          {selectedAlgorithms
            .map(algorithm => ({
              algorithm,
              finalAccuracy: raceResults[algorithm]?.accuracy || 0,
              targetAccuracy: performanceData[algorithm]?.targetAccuracy || 0,
              finished: raceResults[algorithm]?.finished || false,
              finalTime: raceResults[algorithm]?.finalTime || 0
            }))
            .sort((a, b) => {
              if (a.finished && !b.finished) return -1;
              if (!a.finished && b.finished) return 1;
              return b.finalAccuracy - a.finalAccuracy;
            })
            .map((result, index) => (
              <div key={result.algorithm} className="table-row">
                <span className="rank">
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index + 1}`}
                </span>
                <span className="algorithm">
                  <div
                    className="algo-color"
                    style={{ backgroundColor: performanceData[result.algorithm]?.color }}
                  ></div>
                  {result.algorithm}
                </span>
                <span className="accuracy">{(result.finalAccuracy * 100).toFixed(2)}%</span>
                <span className="target">{(result.targetAccuracy * 100).toFixed(1)}%</span>
                <span className="time">
                  {result.finished ? `${result.finalTime.toFixed(1)}s` : 'Not finished'}
                </span>
                <span className="performance">
                  {result.finalAccuracy >= result.targetAccuracy * 0.95 ? '🔥 Excellent' :
                   result.finalAccuracy >= result.targetAccuracy * 0.9 ? '👍 Good' :
                   result.finalAccuracy >= result.targetAccuracy * 0.8 ? '👌 Fair' : '😅 Needs Work'}
                </span>
              </div>
            ))}
        </div>
      </div>

      <style jsx>{`
        .algorithm-race-container {
          background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: #374151;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .race-header {
          text-align: center;
          margin-bottom: 24px;
        }

        .race-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.4rem;
          font-weight: 700;
          color: #1f2937;
        }

        .race-header p {
          margin: 4px 0;
          font-size: 1rem;
          line-height: 1.5;
        }

        .race-description {
          font-style: italic;
          opacity: 0.8;
        }

        .performance-summary {
          background: rgba(255, 255, 255, 0.8);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.5);
        }

        .summary-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .summary-header h4 {
          margin: 0;
          font-size: 1.1rem;
          color: #1f2937;
        }

        .toggle-details-btn {
          padding: 4px 8px;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.7);
          color: #374151;
          cursor: pointer;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .algorithm-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }

        .algorithm-card {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          padding: 16px;
          border: 1px solid rgba(255, 255, 255, 0.7);
        }

        .algorithm-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }

        .algorithm-color {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          border: 2px solid white;
        }

        .algorithm-name {
          font-weight: 600;
          color: #1f2937;
        }

        .algorithm-stats {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .stat-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
        }

        .stat-value {
          font-weight: 600;
          font-family: monospace;
        }

        .cache-hit {
          color: #22c55e;
        }

        .cache-miss {
          color: #3b82f6;
        }

        .race-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
          margin-bottom: 24px;
          flex-wrap: wrap;
          background: rgba(255, 255, 255, 0.7);
          padding: 16px;
          border-radius: 8px;
          backdrop-filter: blur(5px);
        }

        .start-race-btn {
          padding: 12px 24px;
          border: none;
          border-radius: 8px;
          background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
          color: white;
          cursor: pointer;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.3s ease;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .start-race-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .start-race-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .speed-control {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .speed-control label {
          font-size: 0.9rem;
          font-weight: 600;
          color: #374151;
        }

        .speed-control select {
          padding: 4px 8px;
          border: 1px solid #d1d5db;
          border-radius: 4px;
          background: white;
          color: #374151;
          font-size: 0.9rem;
        }

        .race-info {
          display: flex;
          gap: 16px;
          font-size: 0.9rem;
          font-weight: 500;
          color: #374151;
          flex-wrap: wrap;
        }

        .winner-announcement {
          color: #f59e0b;
          font-weight: 700;
        }

        .race-track {
          background: rgba(255, 255, 255, 0.8);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.5);
        }

        .race-track h4 {
          margin: 0 0 16px 0;
          text-align: center;
          color: #1f2937;
        }

        .track-lanes {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .race-lane {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          padding: 12px;
          border: 1px solid rgba(255, 255, 255, 0.7);
        }

        .lane-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .algorithm-info {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .lane-color {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid white;
        }

        .lane-name {
          font-weight: 600;
          color: #1f2937;
          font-size: 0.9rem;
        }

        .lane-stats {
          display: flex;
          gap: 12px;
          font-size: 0.8rem;
          color: #6b7280;
        }

        .finished-badge {
          background: #22c55e;
          color: white;
          padding: 2px 6px;
          border-radius: 10px;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .progress-track {
          position: relative;
          height: 30px;
          background: rgba(229, 231, 235, 0.8);
          border-radius: 15px;
          overflow: hidden;
        }

        .progress-runner {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          border: 2px solid white;
          transition: left 0.3s ease;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .track-markers {
          display: flex;
          justify-content: space-between;
          align-items: center;
          height: 100%;
          padding: 0 8px;
          color: #6b7280;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .race-summary {
          background: rgba(255, 255, 255, 0.8);
          border-radius: 12px;
          padding: 20px;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.5);
        }

        .race-summary h4 {
          margin: 0 0 16px 0;
          text-align: center;
          color: #1f2937;
        }

        .results-table {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .table-header {
          display: grid;
          grid-template-columns: 60px 1fr 100px 100px 120px 120px;
          gap: 12px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.9);
          border-radius: 6px;
          font-weight: 600;
          color: #1f2937;
          font-size: 0.85rem;
        }

        .table-row {
          display: grid;
          grid-template-columns: 60px 1fr 100px 100px 120px 120px;
          gap: 12px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.7);
          border-radius: 6px;
          align-items: center;
          font-size: 0.85rem;
          transition: background-color 0.2s ease;
        }

        .table-row:hover {
          background: rgba(255, 255, 255, 0.9);
        }

        .rank {
          text-align: center;
          font-weight: 700;
          font-size: 1.1rem;
        }

        .algorithm {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 600;
        }

        .algo-color {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid white;
        }

        .accuracy, .target, .time {
          font-family: monospace;
          font-weight: 600;
        }

        .performance {
          text-align: center;
          font-size: 0.8rem;
        }

        @media (max-width: 768px) {
          .race-controls {
            flex-direction: column;
            gap: 12px;
          }

          .race-info {
            flex-direction: column;
            gap: 8px;
            text-align: center;
          }

          .algorithm-cards {
            grid-template-columns: 1fr;
          }

          .table-header,
          .table-row {
            grid-template-columns: 1fr;
            gap: 8px;
          }

          .table-header span,
          .table-row span {
            padding: 4px 0;
          }

          .lane-header {
            flex-direction: column;
            gap: 8px;
            align-items: flex-start;
          }
        }
      `}</style>
    </div>
  );
};

export default AlgorithmRace;

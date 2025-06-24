// frontend/src/components/AlgorithmRace.js

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

const AlgorithmRace = ({ selectedAlgorithms, dataset }) => {
  const [raceProgress, setRaceProgress] = useState({});
  const [isRacing, setIsRacing] = useState(false);
  const [raceComplete, setRaceComplete] = useState(false);
  const [winner, setWinner] = useState(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [raceHistory, setRaceHistory] = useState([]);
  const raceIntervalRef = useRef(null);

  const algorithmColors = {
    'Decision Tree': '#22c55e',
    'SVM': '#3b82f6',
    'Support Vector Machine': '#3b82f6',
    'Artificial Neural Network': '#8b5cf6',
    'Neural Network': '#8b5cf6',
    'K-Nearest Neighbor': '#f59e0b',
    'KNN': '#f59e0b',
    'Logistic Regression': '#ef4444',
    'Random Forest': '#06b6d4',
    'Naive Bayes': '#84cc16',
    'Gradient Boosting': '#f97316'
  };

  const algorithmEmojis = {
    'Decision Tree': '🌳',
    'SVM': '⚡',
    'Support Vector Machine': '⚡',
    'Artificial Neural Network': '🧠',
    'Neural Network': '🧠',
    'K-Nearest Neighbor': '🎯',
    'KNN': '🎯',
    'Logistic Regression': '📈',
    'Random Forest': '🌲',
    'Naive Bayes': '🎲',
    'Gradient Boosting': '🚀'
  };

  // Simulate different learning curves for each algorithm
  const generateLearningCurve = (algorithm, totalEpochs = 100) => {
    const curve = [];

    for (let epoch = 0; epoch <= totalEpochs; epoch++) {
      const progress = epoch / totalEpochs;
      let accuracy = 0;

      switch(algorithm) {
        case 'Decision Tree':
          // Fast initial learning, then plateaus
          accuracy = Math.min(0.92, 0.3 + 0.5 * (1 - Math.exp(-progress * 5)) + Math.random() * 0.03);
          break;

        case 'SVM':
        case 'Support Vector Machine':
          // Steady improvement, good final performance
          accuracy = Math.min(0.94, 0.4 + 0.5 * progress + Math.sin(progress * 3) * 0.02);
          break;

        case 'Artificial Neural Network':
        case 'Neural Network':
          // Slow start, then rapid improvement
          accuracy = Math.min(0.96, 0.2 + 0.7 * Math.pow(progress, 2) + Math.random() * 0.02);
          break;

        case 'K-Nearest Neighbor':
        case 'KNN':
          // Quick learning but limited performance
          accuracy = Math.min(0.88, 0.5 + 0.3 * (1 - Math.exp(-progress * 3)) + Math.random() * 0.02);
          break;

        case 'Logistic Regression':
          // Linear improvement
          accuracy = Math.min(0.89, 0.4 + 0.45 * progress + Math.random() * 0.02);
          break;

        case 'Random Forest':
          // Consistent good performance
          accuracy = Math.min(0.93, 0.6 + 0.3 * progress + Math.sin(progress * 2) * 0.01);
          break;

        case 'Naive Bayes':
          // Fast convergence, moderate performance
          accuracy = Math.min(0.85, 0.5 + 0.3 * (1 - Math.exp(-progress * 4)) + Math.random() * 0.02);
          break;

        case 'Gradient Boosting':
          // Excellent final performance, but slower start
          accuracy = Math.min(0.97, 0.3 + 0.6 * Math.pow(progress, 1.5) + Math.random() * 0.01);
          break;

        default:
          accuracy = 0.5 + 0.4 * progress + Math.random() * 0.03;
      }

      curve.push({
        epoch,
        accuracy: Math.max(0, accuracy),
        algorithm
      });
    }

    return curve;
  };

  useEffect(() => {
    // Initialize race data
    const initialProgress = {};
    selectedAlgorithms.forEach(alg => {
      initialProgress[alg] = {
        currentAccuracy: 0.3 + Math.random() * 0.1,
        maxAccuracy: 0,
        learningCurve: generateLearningCurve(alg),
        isFinished: false,
        finalRank: null,
        speed: 0.8 + Math.random() * 0.4 // Different learning speeds
      };
    });
    setRaceProgress(initialProgress);
    setRaceHistory([]);
    setCurrentEpoch(0);
    setRaceComplete(false);
    setWinner(null);
  }, [selectedAlgorithms]);

  const startRace = () => {
    if (isRacing) return;

    setIsRacing(true);
    setRaceComplete(false);
    setCurrentEpoch(0);
    setRaceHistory([]);

    raceIntervalRef.current = setInterval(() => {
      setCurrentEpoch(prev => {
        const newEpoch = prev + 1;

        setRaceProgress(prevProgress => {
          const newProgress = { ...prevProgress };
          let allFinished = true;
          const currentRankings = [];

          selectedAlgorithms.forEach(alg => {
            const curve = newProgress[alg].learningCurve;
            const currentPoint = curve[Math.min(newEpoch, curve.length - 1)];

            if (currentPoint) {
              newProgress[alg] = {
                ...newProgress[alg],
                currentAccuracy: currentPoint.accuracy,
                maxAccuracy: Math.max(newProgress[alg].maxAccuracy, currentPoint.accuracy)
              };

              currentRankings.push({
                algorithm: alg,
                accuracy: currentPoint.accuracy
              });
            }

            if (newEpoch < curve.length - 1) {
              allFinished = false;
            }
          });

          // Update rankings
          currentRankings.sort((a, b) => b.accuracy - a.accuracy);

          if (allFinished) {
            setIsRacing(false);
            setRaceComplete(true);
            setWinner(currentRankings[0]);
            clearInterval(raceIntervalRef.current);

            // Set final ranks
            currentRankings.forEach((item, index) => {
              newProgress[item.algorithm].finalRank = index + 1;
            });
          }

          return newProgress;
        });

        // Store history for chart
        setRaceHistory(prev => {
          const newHistory = [...prev];
          const historyPoint = { epoch: newEpoch };
          selectedAlgorithms.forEach(alg => {
            const curve = raceProgress[alg]?.learningCurve;
            if (curve && curve[newEpoch]) {
              historyPoint[alg] = curve[newEpoch].accuracy;
            }
          });
          newHistory.push(historyPoint);
          return newHistory;
        });

        return newEpoch >= 100 ? 100 : newEpoch;
      });
    }, 150); // Fast animation
  };

  const stopRace = () => {
    setIsRacing(false);
    if (raceIntervalRef.current) {
      clearInterval(raceIntervalRef.current);
    }
  };

  const resetRace = () => {
    stopRace();
    setCurrentEpoch(0);
    setRaceHistory([]);
    setRaceComplete(false);
    setWinner(null);

    // Reset progress
    const resetProgress = {};
    selectedAlgorithms.forEach(alg => {
      resetProgress[alg] = {
        ...raceProgress[alg],
        currentAccuracy: 0.3 + Math.random() * 0.1,
        maxAccuracy: 0,
        isFinished: false,
        finalRank: null
      };
    });
    setRaceProgress(resetProgress);
  };

  useEffect(() => {
    return () => {
      if (raceIntervalRef.current) {
        clearInterval(raceIntervalRef.current);
      }
    };
  }, []);

  // Prepare chart data
  const chartData = {
    labels: raceHistory.map(h => h.epoch),
    datasets: selectedAlgorithms.map(alg => ({
      label: alg,
      data: raceHistory.map(h => h[alg] || 0),
      borderColor: algorithmColors[alg] || '#6b7280',
      backgroundColor: (algorithmColors[alg] || '#6b7280') + '20',
      borderWidth: 3,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 5
    }))
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        display: true,
        labels: {
          color: '#374151',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        text: `Algorithm Learning Race - Epoch ${currentEpoch}/100`,
        color: '#374151',
        font: { size: 14, weight: 'bold' }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'Training Epoch', color: '#374151' },
        ticks: { color: '#6b7280' },
        grid: { color: 'rgba(107, 114, 128, 0.1)' }
      },
      y: {
        title: { display: true, text: 'Accuracy', color: '#374151' },
        min: 0,
        max: 1,
        ticks: { color: '#6b7280' },
        grid: { color: 'rgba(107, 114, 128, 0.1)' }
      }
    },
    animation: {
      duration: 0 // Real-time updates
    }
  };

  // Get current rankings
  const getCurrentRankings = () => {
    return selectedAlgorithms
      .map(alg => ({
        algorithm: alg,
        accuracy: raceProgress[alg]?.currentAccuracy || 0,
        maxAccuracy: raceProgress[alg]?.maxAccuracy || 0
      }))
      .sort((a, b) => b.accuracy - a.accuracy);
  };

  return (
    <div className="algorithm-race-container">
      <div className="race-header">
        <h3>🏁 Algorithm Learning Race</h3>
        <p>Watch different algorithms compete to learn the {dataset} dataset!</p>
      </div>

      <div className="race-controls">
        <button
          onClick={startRace}
          disabled={isRacing || selectedAlgorithms.length < 2}
          className="control-btn start-btn"
        >
          {isRacing ? '🏃 Racing...' : '🚀 Start Race!'}
        </button>

        <button
          onClick={stopRace}
          disabled={!isRacing}
          className="control-btn stop-btn"
        >
          ⏹️ Stop
        </button>

        <button
          onClick={resetRace}
          className="control-btn reset-btn"
        >
          🔄 Reset Race
        </button>

        {raceComplete && winner && (
          <div className="winner-announcement">
            🏆 Winner: <strong>{winner.algorithm}</strong> with {winner.accuracy.toFixed(4)} accuracy!
          </div>
        )}
      </div>

      <div className="race-track">
        <h4>🏁 Current Standings</h4>
        <div className="rankings">
          {getCurrentRankings().map((item, index) => (
            <div
              key={item.algorithm}
              className={`ranking-item ${index === 0 ? 'leader' : ''}`}
            >
              <div className="rank-position">#{index + 1}</div>
              <div className="algorithm-info">
                <span className="emoji">{algorithmEmojis[item.algorithm] || '🤖'}</span>
                <span className="name">{item.algorithm}</span>
              </div>
              <div className="progress-section">
                <div className="accuracy-display">
                  {item.accuracy.toFixed(4)}
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${item.accuracy * 100}%`,
                      backgroundColor: algorithmColors[item.algorithm] || '#6b7280'
                    }}
                  ></div>
                </div>
              </div>
              {raceComplete && raceProgress[item.algorithm]?.finalRank && (
                <div className="final-rank">
                  Final: #{raceProgress[item.algorithm].finalRank}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="chart-container" style={{ height: '400px' }}>
        <Line data={chartData} options={chartOptions} />
      </div>

      <div className="race-stats">
        <div className="stat-item">
          <span className="label">Current Epoch:</span>
          <span className="value">{currentEpoch}/100</span>
        </div>
        <div className="stat-item">
          <span className="label">Competitors:</span>
          <span className="value">{selectedAlgorithms.length}</span>
        </div>
        <div className="stat-item">
          <span className="label">Race Status:</span>
          <span className="value">
            {raceComplete ? '🏁 Finished' : isRacing ? '🏃 Running' : '⏸️ Ready'}
          </span>
        </div>
      </div>

      <style jsx>{`
        .algorithm-race-container {
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: white;
        }

        .race-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .race-header h3 {
          margin: 0 0 8px 0;
          font-size: 1.5rem;
        }

        .race-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 16px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .control-btn {
          padding: 10px 20px;
          border: none;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-weight: 600;
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

        .winner-announcement {
          background: rgba(255, 215, 0, 0.3);
          padding: 8px 16px;
          border-radius: 20px;
          font-size: 1.1rem;
          font-weight: 600;
          border: 2px solid rgba(255, 215, 0, 0.8);
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }

        .race-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          backdrop-filter: blur(10px);
        }

        .race-track h4 {
          margin: 0 0 16px 0;
          text-align: center;
        }

        .rankings {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .ranking-item {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          transition: all 0.3s ease;
        }

        .ranking-item.leader {
          background: rgba(255, 215, 0, 0.3);
          border: 2px solid rgba(255, 215, 0, 0.8);
          box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }

        .rank-position {
          font-size: 1.2rem;
          font-weight: bold;
          min-width: 40px;
          text-align: center;
        }

        .algorithm-info {
          display: flex;
          align-items: center;
          gap: 8px;
          min-width: 200px;
        }

        .emoji {
          font-size: 1.2rem;
        }

        .name {
          font-weight: 500;
          font-size: 0.9rem;
        }

        .progress-section {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .accuracy-display {
          font-weight: 600;
          min-width: 60px;
          text-align: right;
          font-size: 0.9rem;
        }

        .progress-track {
          flex: 1;
          height: 8px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          transition: width 0.3s ease;
          border-radius: 4px;
        }

        .final-rank {
          font-size: 0.8rem;
          opacity: 0.8;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .race-stats {
          display: flex;
          justify-content: space-around;
          align-items: center;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(10px);
        }

        .stat-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .label {
          font-size: 0.8rem;
          opacity: 0.8;
        }

        .value {
          font-size: 1rem;
          font-weight: 600;
        }

        @media (max-width: 768px) {
          .ranking-item {
            flex-direction: column;
            gap: 8px;
          }

          .algorithm-info {
            min-width: auto;
          }

          .progress-section {
            width: 100%;
          }

          .race-stats {
            flex-direction: column;
            gap: 12px;
          }

          .race-controls {
            flex-direction: column;
            gap: 12px;
          }
        }
      `}</style>
    </div>
  );
};

export default AlgorithmRace;

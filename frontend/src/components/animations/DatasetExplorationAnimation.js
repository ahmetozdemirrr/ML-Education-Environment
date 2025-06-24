import React, { useState, useEffect, useRef } from 'react';
import { Scatter } from 'react-chartjs-2';

const DatasetExplorationAnimation = ({ dataset, algorithm }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dataPoints, setDataPoints] = useState([]);
  const [explorationSteps, setExplorationSteps] = useState([]);
  
  const steps = [
    { name: 'Raw Data Loading', description: 'Loading dataset points...' },
    { name: 'Feature Scaling', description: 'Normalizing feature values...' },
    { name: 'Outlier Detection', description: 'Identifying unusual data points...' },
    { name: 'Class Distribution', description: 'Analyzing class balance...' },
    { name: 'Feature Correlation', description: 'Finding feature relationships...' },
    { name: 'Train/Test Split', description: 'Dividing data for validation...' }
  ];

  // Generate synthetic data based on dataset type
  const generateDataPoints = (step) => {
    const points = [];
    const numPoints = 200;
    
    for (let i = 0; i < numPoints; i++) {
      let x, y, color, size;
      
      switch(step) {
        case 0: // Raw Data
          x = Math.random() * 10 - 5;
          y = Math.random() * 10 - 5;
          color = '#94a3b8';
          size = 4;
          break;
          
        case 1: // Feature Scaling
          x = (Math.random() - 0.5) * 4; // Normalized
          y = (Math.random() - 0.5) * 4;
          color = '#3b82f6';
          size = 4;
          break;
          
        case 2: // Outlier Detection
          if (i < 10) { // Outliers
            x = Math.random() * 8 - 4;
            y = Math.random() * 8 - 4;
            color = '#ef4444';
            size = 8;
          } else {
            x = (Math.random() - 0.5) * 3;
            y = (Math.random() - 0.5) * 3;
            color = '#3b82f6';
            size = 4;
          }
          break;
          
        case 3: // Class Distribution
          const classNum = i % 3;
          x = Math.random() * 2 - 1 + classNum * 2;
          y = Math.random() * 2 - 1 + Math.sin(classNum) * 2;
          color = ['#ef4444', '#22c55e', '#8b5cf6'][classNum];
          size = 5;
          break;
          
        case 4: // Feature Correlation
          x = Math.random() * 4 - 2;
          y = x * 0.7 + Math.random() * 0.5; // Correlated
          color = '#f59e0b';
          size = 4;
          break;
          
        case 5: // Train/Test Split
          const isTrain = i < numPoints * 0.8;
          x = Math.random() * 4 - 2;
          y = Math.random() * 4 - 2;
          color = isTrain ? '#22c55e' : '#ef4444';
          size = isTrain ? 4 : 6;
          break;
          
        default:
          x = y = 0;
          color = '#94a3b8';
          size = 4;
      }
      
      points.push({ x, y, color, size, type: step < 5 ? 'data' : (i < numPoints * 0.8 ? 'train' : 'test') });
    }
    
    return points;
  };

  useEffect(() => {
    setDataPoints(generateDataPoints(currentStep));
  }, [currentStep]);

  const startAnimation = () => {
    setIsPlaying(true);
    setCurrentStep(0);
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          setIsPlaying(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 2000);
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
        text: `${steps[currentStep]?.name} - ${dataset}`,
        font: { size: 16, weight: 'bold' }
      }
    },
    scales: {
      x: { min: -6, max: 6, title: { display: true, text: 'Feature 1' }},
      y: { min: -6, max: 6, title: { display: true, text: 'Feature 2' }}
    }
  };

  return (
    <div className="dataset-exploration-container">
      <div className="exploration-header">
        <h3>📊 Dataset Exploration Journey</h3>
        <p>Watch how we prepare data for {algorithm}</p>
      </div>

      <div className="exploration-controls">
        <button onClick={startAnimation} disabled={isPlaying}>
          {isPlaying ? '🔄 Exploring...' : '🚀 Start Exploration'}
        </button>
        <div className="step-indicator">
          Step {currentStep + 1} of {steps.length}: {steps[currentStep]?.name}
        </div>
      </div>

      <div className="exploration-description">
        <p>{steps[currentStep]?.description}</p>
        
        {currentStep === 2 && (
          <div className="outlier-info">
            <span style={{color: '#ef4444'}}>🔴 Outliers detected</span> |
            <span style={{color: '#3b82f6'}}> 🔵 Normal points</span>
          </div>
        )}
        
        {currentStep === 3 && (
          <div className="class-info">
            <span style={{color: '#ef4444'}}>🔴 Class A</span> |
            <span style={{color: '#22c55e'}}> 🟢 Class B</span> |
            <span style={{color: '#8b5cf6'}}> 🟣 Class C</span>
          </div>
        )}
        
        {currentStep === 5 && (
          <div className="split-info">
            <span style={{color: '#22c55e'}}>🟢 Training Set (80%)</span> |
            <span style={{color: '#ef4444'}}> 🔴 Test Set (20%)</span>
          </div>
        )}
      </div>

      <div className="chart-container" style={{ height: '400px' }}>
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
          </div>
        ))}
      </div>

      <style jsx>{`
        .dataset-exploration-container {
          background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: #8b4513;
        }

        .exploration-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .exploration-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 16px;
        }

        .exploration-controls button {
          padding: 10px 20px;
          border: none;
          border-radius: 8px;
          background: rgba(139, 69, 19, 0.8);
          color: white;
          cursor: pointer;
          font-weight: 600;
        }

        .exploration-controls button:hover:not(:disabled) {
          background: rgba(139, 69, 19, 1);
          transform: translateY(-2px);
        }

        .step-indicator {
          background: rgba(255, 255, 255, 0.8);
          padding: 8px 16px;
          border-radius: 20px;
          font-weight: 600;
        }

        .exploration-description {
          background: rgba(255, 255, 255, 0.6);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
          text-align: center;
        }

        .chart-container {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .progress-steps {
          display: flex;
          justify-content: space-between;
          margin-top: 20px;
          flex-wrap: wrap;
          gap: 8px;
        }

        .progress-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 8px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.3);
          transition: all 0.3s ease;
          flex: 1;
          min-width: 120px;
        }

        .progress-step.completed {
          background: rgba(34, 197, 94, 0.3);
          color: #15803d;
        }

        .progress-step.active {
          background: rgba(59, 130, 246, 0.3);
          color: #1d4ed8;
          transform: scale(1.05);
        }

        .step-number {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: currentColor;
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: bold;
          margin-bottom: 4px;
        }

        .step-name {
          font-size: 0.8rem;
          text-align: center;
          font-weight: 500;
        }

        @media (max-width: 768px) {
          .progress-steps {
            flex-direction: column;
          }
          
          .progress-step {
            flex-direction: row;
            gap: 12px;
          }
        }
      `}</style>
    </div>
  );
};

export default DatasetExplorationAnimation;
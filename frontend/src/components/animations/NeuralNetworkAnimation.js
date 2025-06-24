import React, { useState, useEffect, useRef } from 'react';

const NeuralNetworkAnimation = ({ algorithm, layers = [4, 6, 4, 2] }) => {
  const [activations, setActivations] = useState([]);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [weights, setWeights] = useState([]);
  const canvasRef = useRef(null);

  // Initialize network
  useEffect(() => {
    initializeNetwork();
  }, [layers]);

  const initializeNetwork = () => {
    // Initialize activations for each layer
    const newActivations = layers.map(layerSize => 
      Array(layerSize).fill(0).map(() => Math.random())
    );
    setActivations(newActivations);

    // Initialize weights between layers
    const newWeights = [];
    for (let i = 0; i < layers.length - 1; i++) {
      const layerWeights = [];
      for (let j = 0; j < layers[i]; j++) {
        for (let k = 0; k < layers[i + 1]; k++) {
          layerWeights.push({
            from: { layer: i, node: j },
            to: { layer: i + 1, node: k },
            weight: (Math.random() - 0.5) * 2,
            active: false
          });
        }
      }
      newWeights.push(...layerWeights);
    }
    setWeights(newWeights);
  };

  const startForwardPass = () => {
    setIsTraining(true);
    setCurrentLayer(0);

    const interval = setInterval(() => {
      setCurrentLayer(prev => {
        if (prev >= layers.length - 1) {
          setIsTraining(false);
          clearInterval(interval);
          return prev;
        }

        // Simulate forward propagation
        setActivations(prevActivations => {
          const newActivations = [...prevActivations];
          if (prev + 1 < layers.length) {
            newActivations[prev + 1] = newActivations[prev + 1].map(() => 
              Math.max(0, Math.random() * 2 - 0.5) // ReLU-like activation
            );
          }
          return newActivations;
        });

        // Activate connections
        setWeights(prevWeights => 
          prevWeights.map(w => ({
            ...w,
            active: w.from.layer === prev && w.to.layer === prev + 1
          }))
        );

        return prev + 1;
      });
    }, 1000);
  };

  const drawNetwork = (canvas, ctx) => {
    if (!canvas || !ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const layerSpacing = width / (layers.length + 1);
    const nodePositions = [];

    // Calculate node positions
    layers.forEach((layerSize, layerIndex) => {
      const nodeSpacing = height / (layerSize + 1);
      const layerNodes = [];
      
      for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
        layerNodes.push({
          x: layerSpacing * (layerIndex + 1),
          y: nodeSpacing * (nodeIndex + 1),
          activation: activations[layerIndex]?.[nodeIndex] || 0
        });
      }
      nodePositions.push(layerNodes);
    });

    // Draw connections
    weights.forEach(weight => {
      const fromPos = nodePositions[weight.from.layer]?.[weight.from.node];
      const toPos = nodePositions[weight.to.layer]?.[weight.to.node];
      
      if (fromPos && toPos) {
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        ctx.lineTo(toPos.x, toPos.y);
        
        if (weight.active) {
          ctx.strokeStyle = weight.weight > 0 ? '#22c55e' : '#ef4444';
          ctx.lineWidth = Math.abs(weight.weight) * 2 + 1;
          ctx.globalAlpha = 0.8;
        } else {
          ctx.strokeStyle = '#e5e7eb';
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.3;
        }
        
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // Draw nodes
    nodePositions.forEach((layer, layerIndex) => {
      layer.forEach((node, nodeIndex) => {
        const isActive = layerIndex <= currentLayer;
        const radius = 15;
        
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        
        if (isActive) {
          const intensity = node.activation;
          ctx.fillStyle = `rgba(59, 130, 246, ${0.3 + intensity * 0.7})`;
          ctx.strokeStyle = '#1d4ed8';
          ctx.lineWidth = 2;
        } else {
          ctx.fillStyle = '#f3f4f6';
          ctx.strokeStyle = '#d1d5db';
          ctx.lineWidth = 1;
        }
        
        ctx.fill();
        ctx.stroke();

        // Node label
        ctx.fillStyle = '#374151';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(
          isActive ? node.activation.toFixed(2) : '0.00',
          node.x,
          node.y + 3
        );
      });
    });

    // Layer labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    const layerNames = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
    layers.forEach((_, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      ctx.fillText(
        layerNames[layerIndex] || `Layer ${layerIndex + 1}`,
        x,
        20
      );
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 600;
    canvas.height = 300;

    drawNetwork(canvas, ctx);
  }, [activations, currentLayer, weights, layers]);

  return (
    <div className="neural-network-container">
      <div className="network-header">
        <h3>🧠 Neural Network Forward Pass</h3>
        <p>Watch how {algorithm} processes information through layers</p>
      </div>

      <div className="network-controls">
        <button 
          onClick={startForwardPass}
          disabled={isTraining}
          className="forward-pass-btn"
        >
          {isTraining ? '🔄 Processing...' : '⚡ Start Forward Pass'}
        </button>
        
        <button 
          onClick={initializeNetwork}
          disabled={isTraining}
          className="reset-network-btn"
        >
          🎲 Randomize Weights
        </button>

        <div className="current-layer-indicator">
          Processing Layer: {currentLayer + 1} / {layers.length}
        </div>
      </div>

      <div className="network-canvas-container">
        <canvas ref={canvasRef} className="network-canvas" />
      </div>

      <div className="network-legend">
        <div className="legend-item">
          <div className="legend-color active-node"></div>
          <span>Active Neuron</span>
        </div>
        <div className="legend-item">
          <div className="legend-color inactive-node"></div>
          <span>Inactive Neuron</span>
        </div>
        <div className="legend-item">
          <div className="legend-line positive-weight"></div>
          <span>Positive Weight</span>
        </div>
        <div className="legend-item">
          <div className="legend-line negative-weight"></div>
          <span>Negative Weight</span>
        </div>
      </div>

      <div className="network-info">
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Architecture:</span>
            <span className="info-value">{layers.join(' → ')}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Total Parameters:</span>
            <span className="info-value">
              {layers.reduce((total, curr, i) => 
                i === 0 ? 0 : total + (layers[i-1] * curr), 0
              )}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">Activation Function:</span>
            <span className="info-value">ReLU</span>
          </div>
        </div>
      </div>

      <style jsx>{`
        .neural-network-container {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: white;
        }

        .network-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .network-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 16px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .forward-pass-btn, .reset-network-btn {
          padding: 10px 20px;
          border: none;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s ease;
        }

        .forward-pass-btn:hover:not(:disabled), 
        .reset-network-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
          transform: translateY(-2px);
        }

        .forward-pass-btn:disabled, 
        .reset-network-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .current-layer-indicator {
          background: rgba(255, 255, 255, 0.2);
          padding: 8px 16px;
          border-radius: 20px;
          font-size: 0.9rem;
        }

        .network-canvas-container {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: center;
        }

        .network-canvas {
          border-radius: 4px;
          max-width: 100%;
          height: auto;
        }

        .network-legend {
          display: flex;
          justify-content: center;
          gap: 24px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.9rem;
        }

        .legend-color {
          width: 16px;
          height: 16px;
          border-radius: 50%;
        }

        .legend-color.active-node {
          background: rgba(59, 130, 246, 0.8);
          border: 2px solid #1d4ed8;
        }

        .legend-color.inactive-node {
          background: #f3f4f6;
          border: 2px solid #d1d5db;
        }

        .legend-line {
          width: 20px;
          height: 3px;
        }

        .legend-line.positive-weight {
          background: #22c55e;
        }

        .legend-line.negative-weight {
          background: #ef4444;
        }

        .network-info {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
        }

        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }

        .info-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .info-label {
          font-size: 0.9rem;
          opacity: 0.9;
        }

        .info-value {
          font-weight: 600;
          font-family: monospace;
        }

        @media (max-width: 768px) {
          .network-controls {
            flex-direction: column;
            gap: 12px;
          }
          
          .network-legend {
            flex-direction: column;
            align-items: center;
            gap: 12px;
          }
          
          .info-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default NeuralNetworkAnimation;
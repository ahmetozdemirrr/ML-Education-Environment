import React, { useState, useEffect, useRef } from 'react';

const DecisionTreeAnimation = ({ algorithm, dataset }) => {
  const [treeNodes, setTreeNodes] = useState([]);
  const [currentDepth, setCurrentDepth] = useState(0);
  const [isGrowing, setIsGrowing] = useState(false);
  const [splitCriteria, setSplitCriteria] = useState([]);
  const treeCanvasRef = useRef(null);

  const treeStructure = [
    { 
      id: 0, depth: 0, parent: null, 
      condition: 'Root Node', 
      samples: 1000, 
      gini: 0.5, 
      feature: 'Start',
      position: { x: 300, y: 50 }
    },
    { 
      id: 1, depth: 1, parent: 0, 
      condition: 'Feature_A ≤ 2.5', 
      samples: 600, 
      gini: 0.3, 
      feature: 'Feature_A',
      position: { x: 200, y: 150 }
    },
    { 
      id: 2, depth: 1, parent: 0, 
      condition: 'Feature_A > 2.5', 
      samples: 400, 
      gini: 0.4, 
      feature: 'Feature_A',
      position: { x: 400, y: 150 }
    },
    { 
      id: 3, depth: 2, parent: 1, 
      condition: 'Feature_B ≤ 1.8', 
      samples: 350, 
      gini: 0.1, 
      feature: 'Feature_B',
      position: { x: 150, y: 250 }
    },
    { 
      id: 4, depth: 2, parent: 1, 
      condition: 'Feature_B > 1.8', 
      samples: 250, 
      gini: 0.2, 
      feature: 'Feature_B',
      position: { x: 250, y: 250 }
    },
    { 
      id: 5, depth: 2, parent: 2, 
      condition: 'Feature_C ≤ 0.6', 
      samples: 200, 
      gini: 0.0, 
      feature: 'Feature_C',
      position: { x: 350, y: 250 }
    },
    { 
      id: 6, depth: 2, parent: 2, 
      condition: 'Feature_C > 0.6', 
      samples: 200, 
      gini: 0.0, 
      feature: 'Feature_C',
      position: { x: 450, y: 250 }
    }
  ];

  const maxDepth = Math.max(...treeStructure.map(node => node.depth));

  const startGrowth = () => {
    setIsGrowing(true);
    setCurrentDepth(0);
    setTreeNodes([]);
    setSplitCriteria([]);

    const interval = setInterval(() => {
      setCurrentDepth(prev => {
        if (prev >= maxDepth) {
          setIsGrowing(false);
          clearInterval(interval);
          return prev;
        }

        // Add nodes at current depth
        const nodesAtDepth = treeStructure.filter(node => node.depth === prev + 1);
        setTreeNodes(prevNodes => [...prevNodes, ...nodesAtDepth]);
        
        // Add split criteria
        if (prev < maxDepth) {
          setSplitCriteria(prevCriteria => [
            ...prevCriteria,
            {
              depth: prev + 1,
              feature: nodesAtDepth[0]?.feature,
              giniReduction: 0.5 - Math.random() * 0.3
            }
          ]);
        }

        return prev + 1;
      });
    }, 1500);
  };

  const drawTree = (canvas, ctx) => {
    if (!canvas || !ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const visibleNodes = treeStructure.filter(node => node.depth <= currentDepth);

    // Draw connections
    visibleNodes.forEach(node => {
      if (node.parent !== null) {
        const parent = visibleNodes.find(n => n.id === node.parent);
        if (parent) {
          ctx.beginPath();
          ctx.moveTo(parent.position.x, parent.position.y + 30);
          ctx.lineTo(node.position.x, node.position.y - 30);
          ctx.strokeStyle = '#6b7280';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    });

    // Draw nodes
    visibleNodes.forEach(node => {
      const nodeWidth = 120;
      const nodeHeight = 60;
      const x = node.position.x - nodeWidth / 2;
      const y = node.position.y - nodeHeight / 2;

      // Node background (color based on gini impurity)
      const giniColor = `hsl(${(1 - node.gini) * 120}, 70%, 85%)`;
      ctx.fillStyle = giniColor;
      ctx.fillRect(x, y, nodeWidth, nodeHeight);

      // Node border
      ctx.strokeStyle = node.depth === currentDepth ? '#2563eb' : '#6b7280';
      ctx.lineWidth = node.depth === currentDepth ? 3 : 1;
      ctx.strokeRect(x, y, nodeWidth, nodeHeight);

      // Node text
      ctx.fillStyle = '#1f2937';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      
      // Condition
      ctx.font = 'bold 10px Arial';
      ctx.fillText(node.condition, node.position.x, node.position.y - 10);
      
      // Samples
      ctx.font = '9px Arial';
      ctx.fillText(`Samples: ${node.samples}`, node.position.x, node.position.y + 5);
      
      // Gini
      ctx.fillText(`Gini: ${node.gini.toFixed(3)}`, node.position.x, node.position.y + 18);
    });

    // Draw depth indicator
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Tree Depth: ${currentDepth}`, 10, 25);
  };

  useEffect(() => {
    const canvas = treeCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 600;
    canvas.height = 350;

    drawTree(canvas, ctx);
  }, [currentDepth]);

  return (
    <div className="decision-tree-container">
      <div className="tree-header">
        <h3>🌳 Decision Tree Growth Simulation</h3>
        <p>Watch how {algorithm} builds decision boundaries for {dataset}</p>
      </div>

      <div className="tree-controls">
        <button 
          onClick={startGrowth}
          disabled={isGrowing}
          className="grow-tree-btn"
        >
          {isGrowing ? '🌱 Growing...' : '🚀 Grow Tree'}
        </button>

        <div className="growth-info">
          <span>Current Depth: {currentDepth} / {maxDepth}</span>
          <span>Nodes Created: {treeNodes.length}</span>
        </div>
      </div>

      <div className="tree-canvas-container">
        <canvas ref={treeCanvasRef} className="tree-canvas" />
      </div>

      <div className="split-criteria">
        <h4>📊 Split Criteria Progress</h4>
        <div className="criteria-list">
          {splitCriteria.map((criteria, index) => (
            <div key={index} className="criteria-item">
              <div className="criteria-depth">Depth {criteria.depth}</div>
              <div className="criteria-feature">Feature: {criteria.feature}</div>
              <div className="criteria-reduction">
                Gini Reduction: {criteria.giniReduction.toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="tree-metrics">
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Max Depth:</span>
            <span className="metric-value">{maxDepth}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Total Nodes:</span>
            <span className="metric-value">{treeStructure.length}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Leaf Nodes:</span>
            <span className="metric-value">
              {treeStructure.filter(node => 
                !treeStructure.some(child => child.parent === node.id)
              ).length}
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Features Used:</span>
            <span className="metric-value">
              {[...new Set(treeStructure.map(node => node.feature))].length - 1}
            </span>
          </div>
        </div>
      </div>

      <div className="gini-legend">
        <h4>🎨 Gini Impurity Color Scale</h4>
        <div className="color-scale">
          <div className="scale-item">
            <div className="color-box" style={{backgroundColor: 'hsl(120, 70%, 85%)'}}></div>
            <span>Pure (Gini = 0.0)</span>
          </div>
          <div className="scale-item">
            <div className="color-box" style={{backgroundColor: 'hsl(60, 70%, 85%)'}}></div>
            <span>Mixed (Gini = 0.5)</span>
          </div>
          <div className="scale-item">
            <div className="color-box" style={{backgroundColor: 'hsl(0, 70%, 85%)'}}></div>
            <span>Impure (Gini = 1.0)</span>
          </div>
        </div>
      </div>

      <style jsx>{`
        .decision-tree-container {
          background: linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          color: #065f46;
        }

        .tree-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .tree-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 16px;
        }

        .grow-tree-btn {
          padding: 10px 20px;
          border: none;
          border-radius: 8px;
          background: rgba(6, 95, 70, 0.8);
          color: white;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s ease;
        }

        .grow-tree-btn:hover:not(:disabled) {
          background: rgba(6, 95, 70, 1);
          transform: translateY(-2px);
        }

        .grow-tree-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .growth-info {
          display: flex;
          gap: 20px;
          font-weight: 600;
        }

        .tree-canvas-container {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: center;
        }

        .tree-canvas {
          border-radius: 4px;
          max-width: 100%;
          height: auto;
        }

        .split-criteria {
          background: rgba(255, 255, 255, 0.6);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .split-criteria h4 {
          margin: 0 0 12px 0;
          color: #065f46;
        }

        .criteria-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .criteria-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.8);
          border-radius: 6px;
          font-size: 0.9rem;
        }

        .criteria-depth {
          font-weight: 600;
          color: #059669;
        }

        .tree-metrics {
          background: rgba(255, 255, 255, 0.6);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 16px;
        }

        .metric-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px;
          background: rgba(255, 255, 255, 0.8);
          border-radius: 6px;
        }

        .metric-label {
          font-size: 0.9rem;
          color: #065f46;
        }

        .metric-value {
          font-weight: 600;
          font-family: monospace;
          color: #059669;
        }

        .gini-legend {
          background: rgba(255, 255, 255, 0.6);
          border-radius: 8px;
          padding: 16px;
        }

        .gini-legend h4 {
          margin: 0 0 12px 0;
          color: #065f46;
        }

        .color-scale {
          display: flex;
          justify-content: space-around;
          flex-wrap: wrap;
          gap: 16px;
        }

        .scale-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.9rem;
        }

        .color-box {
          width: 20px;
          height: 20px;
          border-radius: 4px;
          border: 1px solid #065f46;
        }

        @media (max-width: 768px) {
          .tree-controls {
            flex-direction: column;
            align-items: center;
          }
          
          .growth-info {
            flex-direction: column;
            text-align: center;
            gap: 8px;
          }
          
          .criteria-item {
            flex-direction: column;
            gap: 4px;
          }
          
          .color-scale {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
};

export default DecisionTreeAnimation;
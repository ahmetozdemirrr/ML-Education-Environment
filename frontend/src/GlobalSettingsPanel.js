/* ./frontend/src/GlobalSettingsPanel.js */

import React from 'react';
import './App.css';

function GlobalSettingsPanel({ settings, onChange }) {
  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    onChange(name, checked);

    if (name === 'useCrossValidation' && checked) {
      onChange('useTrainTestSplit', false);
    }
    if (name === 'useTrainTestSplit' && checked) {
      onChange('useCrossValidation', false);
    }
  };

  const handleValueChange = (event) => {
    const { name, value } = event.target;
    if (name === 'randomSeedType') {
      onChange(name, value); // "random" veya "fixed" değerini gönder
    } else {
      const { type, value: val } = event.target;
      let processedValue = val;
      if (type === 'number') {
        processedValue = val === '' ? '' : parseFloat(val);
        if (name === 'cvFolds' && processedValue !== '' && processedValue < 2) processedValue = 2;
        if (name === 'testSplitRatio' && processedValue !== '') {
          if (processedValue < 0.01) processedValue = 0.01;
          if (processedValue > 0.99) processedValue = 0.99;
        }
        if (name === 'randomSeed' && processedValue !== '' && processedValue < 0) processedValue = 0;
        if (isNaN(processedValue)) processedValue = settings[name];
      }
      onChange(name, processedValue);
    }
  };

  return (
    <div className="panel settings-panel">
      <h2>Global Settings</h2>
      <div className="settings-list">
        {/* Cross-Validation Section */}
        <div className="setting-group">
          <label className="setting-group-title">
            <input type="checkbox" name="useCrossValidation" checked={settings.useCrossValidation} onChange={handleCheckboxChange} />
            <span>Use Cross-Validation</span>
          </label>
          <div className={`setting-item indented ${settings.useCrossValidation ? 'visible' : 'hidden'}`}>
            <label htmlFor="cvFolds">Number of Folds:</label>
            <input type="number" id="cvFolds" name="cvFolds" value={settings.cvFolds} onChange={handleValueChange} min="2" step="1" disabled={!settings.useCrossValidation} />
          </div>
        </div>

        <div className="setting-group">
          <label className="setting-group-title">
            <input type="checkbox" name="useTrainTestSplit" checked={settings.useTrainTestSplit} onChange={handleCheckboxChange} disabled={settings.useCrossValidation} />
            <span>Use Train/Test Split</span>
          </label>
          <div className={`setting-item indented ${settings.useTrainTestSplit && !settings.useCrossValidation ? 'visible' : 'hidden'}`}>
            <label htmlFor="testSplitRatio">Test Set Ratio (0-1):</label>
            <input type="number" id="testSplitRatio" name="testSplitRatio" value={settings.testSplitRatio} onChange={handleValueChange} min="0.01" max="0.99" step="0.01" disabled={!settings.useTrainTestSplit || settings.useCrossValidation} />
          </div>
        </div>

        <div className="setting-group">
          <label className="setting-group-title full-width">Reproducibility:</label>
          <div className="setting-item indented visible">
            <label>Seed Type:</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="randomSeedType"
                  value="random"
                  checked={settings.randomSeedType === "random"}
                  onChange={handleValueChange}
                />
                Random
              </label>
              <label>
                <input
                  type="radio"
                  name="randomSeedType"
                  value="fixed"
                  checked={settings.randomSeedType === "fixed"}
                  onChange={handleValueChange}
                />
                Fixed (42)
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GlobalSettingsPanel;
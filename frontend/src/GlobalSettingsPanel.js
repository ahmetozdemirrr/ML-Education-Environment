/* ./frontend/src/GlobalSettingsPanel.js - Cross Validation Removed */

import React from 'react';
import './App.css';

function GlobalSettingsPanel({ settings, onChange }) {
  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    onChange(name, checked);
  };

  const handleValueChange = (event) => {
    const { name, value } = event.target;
    if (name === 'randomSeedType') {
      onChange(name, value); // "random" veya "fixed" değerini gönder
    } else if (name === 'scalerType') {
      onChange(name, value); // "standard" veya "minmax" değerini gönder
    } else {
      const { type, value: val } = event.target;
      let processedValue = val;
      if (type === 'number') {
        processedValue = val === '' ? '' : parseFloat(val);
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

        {/* Data Split Settings */}
        <div className="setting-group">
          <div className="setting-group-title full-width">
            <span>Data Split Settings</span>
          </div>

          <div className="setting-item visible">
            <label htmlFor="test-split-ratio">Test Size Ratio:</label>
            <input
              type="number"
              id="test-split-ratio"
              value={settings.testSplitRatio}
              min="0.1"
              max="0.5"
              step="0.05"
              onChange={(e) => onChange('testSplitRatio', parseFloat(e.target.value))}
            />
            <small>Proportion of data</small>
          </div>
        </div>

        {/* Feature Scaling */}
        <div className="setting-group">
          <div className="setting-group-title">
            <input
              type="checkbox"
              id="feature-scaling"
              checked={settings.applyFeatureScaling}
              onChange={(e) => onChange('applyFeatureScaling', e.target.checked)}
            />
            <span>Feature Scaling</span>
          </div>

          <div className={`setting-item indented ${settings.applyFeatureScaling ? 'visible' : 'hidden'}`}>
            <label>Scaler Type:</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="scaler-type"
                  value="standard"
                  checked={settings.scalerType === 'standard'}
                  onChange={(e) => onChange('scalerType', e.target.value)}
                  disabled={!settings.applyFeatureScaling}
                />
                Standard
              </label>
              <label>
                <input
                  type="radio"
                  name="scaler-type"
                  value="minmax"
                  checked={settings.scalerType === 'minmax'}
                  onChange={(e) => onChange('scalerType', e.target.value)}
                  disabled={!settings.applyFeatureScaling}
                />
                MinMax
              </label>
            </div>
          </div>
        </div>

        {/* Reproducibility */}
        <div className="setting-group">
          <div className="setting-group-title full-width">
            <span>Reproducibility</span>
          </div>

          <div className="setting-item visible">
            <label>Random Seed:</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="seed-type"
                  value="fixed"
                  checked={settings.randomSeedType === 'fixed'}
                  onChange={(e) => onChange('randomSeedType', e.target.value)}
                />
                Fixed
              </label>
              <label>
                <input
                  type="radio"
                  name="seed-type"
                  value="random"
                  checked={settings.randomSeedType === 'random'}
                  onChange={(e) => onChange('randomSeedType', e.target.value)}
                />
                Random
              </label>
            </div>
          </div>

          <div className={`setting-item indented ${settings.randomSeedType === 'fixed' ? 'visible' : 'hidden'}`}>
            <label htmlFor="random-seed">Seed Value:</label>
            <input
              type="number"
              id="random-seed"
              value={settings.randomSeed || 42}
              min="0"
              max="9999"
              onChange={(e) => onChange('randomSeed', parseInt(e.target.value, 10))}
              disabled={settings.randomSeedType !== 'fixed'}
            />
            <small>Default: 42</small>
          </div>
        </div>

      </div>
    </div>
  );
}

export default GlobalSettingsPanel;

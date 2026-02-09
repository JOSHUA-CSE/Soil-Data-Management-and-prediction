import React from 'react';
import '../styles/ResultPanel.css';

function ResultPanel({ irrigationRequired, soilHealthIndex }) {
  return (
    <div className="result-panel">
      <div className="result-panel-grid">
        <div className="result-item irrigation-item">
          <div className="result-label">Irrigation Required</div>
          <div className={`irrigation-badge ${irrigationRequired?.toLowerCase() === 'yes' ? 'required' : 'not-required'}`}>
            {irrigationRequired === 'Yes' ? '💧 Yes' : '🌤️ No'}
          </div>
        </div>
        
        <div className="result-item health-item">
          <div className="result-label">Soil Health Index</div>
          <div className="health-container">
            <div className="progress-bar-wrapper">
              <div 
                className="progress-bar-fill" 
                style={{ width: `${Math.min(soilHealthIndex || 0, 100)}%` }}
              ></div>
            </div>
            <div className="health-value">{Math.round(soilHealthIndex || 0)}%</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultPanel;

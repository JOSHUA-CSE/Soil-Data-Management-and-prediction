import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import RecommendationCards from '../components/RecommendationCards';
import ResultPanel from '../components/ResultPanel';
import '../styles/PredictionResult.css';

function PredictionResult() {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction, input } = location.state || {};

  if (!prediction) {
    return (
      <div className="result-container">
        <div className="empty-state">
          <p>No prediction data available.</p>
          <button onClick={() => navigate('/')} className="btn-primary">Go Back</button>
        </div>
      </div>
    );
  }

  return (
    <div className="result-container">
      <div className="result-wrapper">
        <div className="result-header">
          <h2>Crop Analysis Results</h2>
          <p className="header-subtitle">Based on your soil conditions</p>
        </div>

        <RecommendationCards recommendations={prediction.crop_recommendations} />

        <ResultPanel 
          irrigationRequired={prediction.irrigation_required}
          soilHealthIndex={prediction.soil_health_index}
        />

        <div className="input-summary">
          <h3>Input Parameters Used</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="param-name">Soil Moisture:</span>
              <span className="param-value">{input?.soil_moisture || 'N/A'}%</span>
            </div>
            <div className="summary-item">
              <span className="param-name">Soil pH:</span>
              <span className="param-value">{input?.soil_pH || 'N/A'}</span>
            </div>
            <div className="summary-item">
              <span className="param-name">Temperature:</span>
              <span className="param-value">{input?.temperature || 'N/A'}°C</span>
            </div>
            <div className="summary-item">
              <span className="param-name">Rainfall:</span>
              <span className="param-value">{input?.rainfall || 'N/A'} mm</span>
            </div>
            <div className="summary-item">
              <span className="param-name">Humidity:</span>
              <span className="param-value">{input?.humidity || 'N/A'}%</span>
            </div>
          </div>
        </div>

        <div className="action-buttons">
          <button onClick={() => navigate('/')} className="btn-primary">New Prediction</button>
          <button onClick={() => navigate('/history')} className="btn-secondary">View History</button>
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;

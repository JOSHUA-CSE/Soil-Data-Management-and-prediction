import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/PredictionDetail.css';

function PredictionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchPrediction();
  }, [id]);

  const fetchPrediction = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/prediction/${id}`);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to load prediction details');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="detail-container"><div className="loading">Loading...</div></div>;
  }

  if (error || !prediction) {
    return (
      <div className="detail-container">
        <div className="error-message">{error || 'Prediction not found'}</div>
        <button onClick={() => navigate('/history')} className="back-btn">Back to History</button>
      </div>
    );
  }

  return (
    <div className="detail-container">
      <button onClick={() => navigate('/history')} className="back-btn">← Back to History</button>
      
      <h2 className="detail-title">Prediction Details</h2>

      <div className="recommendations-section">
        <h3>Top 3 Recommended Crops</h3>
        <div className="crops-grid">
          {prediction.recommendations?.map((rec, idx) => (
            <div key={idx} className={`crop-card rank-${idx + 1}`}>
              <div className="rank-badge">#{idx + 1}</div>
              <h4>{rec.crop}</h4>
              <div className="crop-detail">
                <span>Expected Yield:</span>
                <strong>{rec.yield} kg/ha</strong>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="irrigation-section">
        <h3>Irrigation Decision</h3>
        <div className={`irrigation-result ${prediction.irrigation === 'Yes' ? 'required' : 'not-required'}`}>
          {prediction.irrigation === 'Yes' ? '💧 Irrigation Required' : '🌤️ No Irrigation Needed'}
        </div>
      </div>

      <div className="explanation-section">
        <h3>Recommendation Explanation</h3>
        <p>{prediction.explanation || 'Based on soil conditions, climate factors, and historical data, these crops are most suitable for your parameters.'}</p>
      </div>

      <div className="input-section">
        <h3>Input Parameters Used</h3>
        <div className="params-grid">
          <div className="param-item"><span>Soil Moisture:</span> {prediction.soil_moisture}%</div>
          <div className="param-item"><span>Soil pH:</span> {prediction.soil_pH}</div>
          <div className="param-item"><span>Temperature:</span> {prediction.temperature}°C</div>
          <div className="param-item"><span>Rainfall:</span> {prediction.rainfall} mm</div>
          <div className="param-item"><span>Humidity:</span> {prediction.humidity}%</div>
        </div>
      </div>
    </div>
  );
}

export default PredictionDetail;

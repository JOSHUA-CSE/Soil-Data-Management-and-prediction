import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { historyAPI } from '../services/api';
import '../styles/History.css';

function History() {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await historyAPI();
      const data = response.data.predictions || [];
      setHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const options = { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' };
    return date.toLocaleDateString('en-GB', options).replace(',', '');
  };

  const handleView = (id) => {
    navigate(`/prediction/${id}`);
  };

  if (loading) {
    return <div className="history-container"><div className="loading">Loading history...</div></div>;
  }

  if (error) {
    return <div className="history-container"><div className="error-message">{error}</div></div>;
  }

  return (
    <div className="history-container">
      <h2 className="history-title">Prediction History</h2>
      <p className="history-subtitle">Input parameters from past predictions</p>
      
      {history.length === 0 ? (
        <div className="empty-state">
          <p>No predictions yet. Start by making your first prediction!</p>
        </div>
      ) : (
        <div className="table-container">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date & Time</th>
                <th>Location</th>
                <th>Soil Moisture (%)</th>
                <th>Soil pH</th>
                <th>Temperature (°C)</th>
                <th>Rainfall (mm)</th>
                <th>Humidity (%)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.id}>
                  <td>{formatDate(item.created_at)}</td>
                  <td>{item.location || `${item.latitude || 'N/A'}, ${item.longitude || 'N/A'}`}</td>
                  <td>{item.soil_moisture}</td>
                  <td>{item.soil_pH}</td>
                  <td>{item.temperature}</td>
                  <td>{item.rainfall}</td>
                  <td>{item.humidity}</td>
                  <td>
                    <button className="view-btn" onClick={() => handleView(item.id)}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default History;

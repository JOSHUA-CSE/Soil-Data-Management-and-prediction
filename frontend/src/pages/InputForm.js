import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictAPI } from '../services/api';
import '../styles/InputForm.css';

function InputForm() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    soil_moisture: '',
    soil_pH: '',
    temperature: '',
    rainfall: '',
    humidity: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const validateForm = () => {
    const { soil_moisture, soil_pH, temperature, rainfall, humidity } = formData;
    
    if (!soil_moisture || !soil_pH || !temperature || !rainfall || !humidity) {
      setError('All fields are required');
      return false;
    }
    
    if (soil_moisture < 0 || soil_moisture > 100) {
      setError('Soil moisture must be between 0-100');
      return false;
    }
    
    if (soil_pH < 0 || soil_pH > 14) {
      setError('Soil pH must be between 0-14');
      return false;
    }
    
    if (temperature < -50 || temperature > 60) {
      setError('Temperature must be between -50 and 60°C');
      return false;
    }
    
    if (rainfall < 0 || rainfall > 5000) {
      setError('Rainfall must be between 0-5000 mm');
      return false;
    }
    
    if (humidity < 0 || humidity > 100) {
      setError('Humidity must be between 0-100');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);

    try {
      const numericData = {
        soil_moisture: parseFloat(formData.soil_moisture),
        soil_pH: parseFloat(formData.soil_pH),
        temperature: parseFloat(formData.temperature),
        rainfall: parseFloat(formData.rainfall),
        humidity: parseFloat(formData.humidity)
      };
      
      const response = await predictAPI(numericData);
      navigate('/result', { state: { prediction: response.data, input: formData } });
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="input-form-container">
      <div className="form-card">
        <div className="form-header">
          <h2 className="form-title">Soil Parameter Analysis</h2>
          <p className="form-subtitle">Enter soil parameters to get crop recommendations</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit} className="form">
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="soil_moisture">Soil Moisture (%)</label>
              <input 
                id="soil_moisture"
                type="number" 
                step="0.01" 
                name="soil_moisture" 
                value={formData.soil_moisture} 
                onChange={handleChange}
                placeholder="0-100"
                disabled={loading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="soil_pH">Soil pH</label>
              <input 
                id="soil_pH"
                type="number" 
                step="0.01" 
                name="soil_pH" 
                value={formData.soil_pH} 
                onChange={handleChange}
                placeholder="0-14"
                disabled={loading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="temperature">Temperature (°C)</label>
              <input 
                id="temperature"
                type="number" 
                step="0.01" 
                name="temperature" 
                value={formData.temperature} 
                onChange={handleChange}
                placeholder="-50 to 60"
                disabled={loading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="rainfall">Rainfall (mm)</label>
              <input 
                id="rainfall"
                type="number" 
                step="0.01" 
                name="rainfall" 
                value={formData.rainfall} 
                onChange={handleChange}
                placeholder="0-5000"
                disabled={loading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="humidity">Humidity (%)</label>
              <input 
                id="humidity"
                type="number" 
                step="0.01" 
                name="humidity" 
                value={formData.humidity} 
                onChange={handleChange}
                placeholder="0-100"
                disabled={loading}
              />
            </div>
          </div>

          <button 
            type="submit" 
            className="submit-btn"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              'Get Recommendations'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default InputForm;

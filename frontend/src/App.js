import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import InputForm from './pages/InputForm';
import PredictionResult from './pages/PredictionResult';
import History from './pages/History';
import PredictionDetail from './pages/PredictionDetail';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="logo">🌾 Smart Agriculture</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">Predict</Link>
              <Link to="/history" className="nav-link">History</Link>
            </div>
          </div>
        </nav>
        <div className="main-content">
          <Routes>
            <Route path="/" element={<InputForm />} />
            <Route path="/result" element={<PredictionResult />} />
            <Route path="/history" element={<History />} />
            <Route path="/prediction/:id" element={<PredictionDetail />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;

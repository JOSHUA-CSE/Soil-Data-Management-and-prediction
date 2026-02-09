import React from 'react';
import '../styles/RecommendationCards.css';

const RANK_BADGES = ['🥇', '🥈', '🥉'];
const RANK_LABELS = ['1st', '2nd', '3rd'];

function RecommendationCards({ recommendations }) {
  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="recommendation-cards-container">
        <p className="no-recommendations">No crop recommendations available</p>
      </div>
    );
  }

  return (
    <div className="recommendation-cards-container">
      <h3 className="recommendations-title">Top Crop Recommendations</h3>
      <div className="cards-grid">
        {recommendations.map((rec, index) => (
          <div 
            key={index} 
            className={`recommendation-card rank-${rec.rank} ${index === 0 ? 'featured' : ''}`}
          >
            <div className="card-header">
              <span className="rank-badge">{RANK_BADGES[index % 3]}</span>
              <span className="rank-label">{RANK_LABELS[index % 3]} Choice</span>
            </div>
            
            <div className="card-body">
              <h4 className="crop-name">{rec.crop}</h4>
              <div className="yield-section">
                <label>Expected Yield</label>
                <p className="yield-value">{rec.predicted_yield?.toFixed(0) || 0} kg/ha</p>
              </div>
            </div>
            
            <div className="card-footer">
              <div className="rank-indicator">Rank {rec.rank}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RecommendationCards;

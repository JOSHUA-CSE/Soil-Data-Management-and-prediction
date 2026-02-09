import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import pandas as pd

def load_models_and_data():
    """Load trained models and test data"""
    models_dir = os.path.join(os.path.dirname(__file__), 'prediction', 'models')
    encoders_dir = os.path.join(os.path.dirname(__file__), 'prediction', 'encoders')
    
    print("Loading models...")
    try:
        yield_model = joblib.load(os.path.join(models_dir, 'yield_predictor.pkl'))
        irrigation_model = joblib.load(os.path.join(models_dir, 'irrigation_model.pkl'))
        
        irrigation_encoder = joblib.load(os.path.join(encoders_dir, 'irrigation_encoder.pkl'))
        metadata = joblib.load(os.path.join(encoders_dir, 'model_metadata.pkl'))
        
        print("Models loaded successfully\n")
        return {
            'yield_model': yield_model,
            'irrigation_model': irrigation_model,
            'irrigation_encoder': irrigation_encoder,
            'metadata': metadata
        }
    except FileNotFoundError as e:
        print(f"Error: Model files not found - {str(e)}")
        print("Please run train_models.py first")
        return None

def generate_test_data(metadata, num_samples=50):
    """Generate synthetic test data using actual irrigation classes"""
    print(f"Generating {num_samples} test samples...\n")
    
    irrigation_classes = metadata.get('irrigation_classes', ['Yes', 'No'])
    
    # Generate base features
    soil_moisture = np.random.uniform(20, 80, num_samples)
    temperature = np.random.uniform(15, 35, num_samples)
    rainfall = np.random.uniform(50, 250, num_samples)
    humidity = np.random.uniform(30, 90, num_samples)
    
    test_data = {
        'soil_moisture': soil_moisture,
        'soil_pH': np.random.uniform(5.5, 8.5, num_samples),
        'temperature': temperature,
        'rainfall': rainfall,
        'humidity': humidity,
        'irrigation': np.random.choice(irrigation_classes, num_samples),
        'yield': np.random.uniform(3000, 6000, num_samples)
    }
    
    # Add engineered features
    test_data['moisture_temp_ratio'] = soil_moisture / (temperature + 1e-6)
    test_data['rainfall_humidity_index'] = rainfall * humidity
    test_data['stress_index'] = temperature / (soil_moisture + 1e-6)
    
    df = pd.DataFrame(test_data)
    return df

def evaluate_models(models, test_data):
    """Evaluate all models and print metrics"""
    
    feature_columns = models['metadata']['feature_columns']
    X_test = pd.DataFrame(test_data[feature_columns], columns=feature_columns)
    
    print("=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70 + "\n")
    
    # Evaluate Yield Predictor
    print("1. YIELD PREDICTOR MODEL")
    print("-" * 70)
    y_true_yield = test_data['yield'].values
    y_pred_yield = models['yield_model'].predict(X_test)
    
    yield_mae = mean_absolute_error(y_true_yield, y_pred_yield)
    yield_rmse = np.sqrt(mean_squared_error(y_true_yield, y_pred_yield))
    yield_mape = np.mean(np.abs((y_true_yield - y_pred_yield) / y_true_yield)) * 100
    
    print(f"Mean Absolute Error (MAE):    {yield_mae:.4f} kg/ha")
    print(f"Root Mean Squared Error (RMSE): {yield_rmse:.4f} kg/ha")
    print(f"Mean Absolute Percentage Error: {yield_mape:.2f}%")
    print()
    
    # Evaluate Irrigation Model
    print("2. IRRIGATION REQUIREMENT MODEL")
    print("-" * 70)
    y_true_irrigation = models['irrigation_encoder'].transform(test_data['irrigation'].values)
    y_pred_irrigation = models['irrigation_model'].predict(X_test)
    
    irrigation_accuracy = accuracy_score(y_true_irrigation, y_pred_irrigation)
    irrigation_precision = precision_score(y_true_irrigation, y_pred_irrigation, average='weighted', zero_division=0)
    irrigation_recall = recall_score(y_true_irrigation, y_pred_irrigation, average='weighted', zero_division=0)
    irrigation_f1 = f1_score(y_true_irrigation, y_pred_irrigation, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {irrigation_accuracy:.4f} ({irrigation_accuracy*100:.2f}%)")
    print(f"Precision: {irrigation_precision:.4f}")
    print(f"Recall:    {irrigation_recall:.4f}")
    print(f"F1-Score:  {irrigation_f1:.4f}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Irrigation Prediction Accuracy:   {irrigation_accuracy*100:6.2f}%")
    print(f"Yield Prediction MAPE:            {yield_mape:6.2f}%")
    print("=" * 70)

def main():
    models = load_models_and_data()
    
    if models is None:
        return
    
    # Generate test data using actual crop/irrigation classes
    test_data = generate_test_data(models['metadata'])
    
    # Evaluate models
    evaluate_models(models, test_data)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()

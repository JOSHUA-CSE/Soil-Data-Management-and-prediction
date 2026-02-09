import joblib
import os
import pandas as pd
from typing import Dict, Any, List, Optional

# CRITICAL: Feature columns must match training data exactly
# Models were trained with pandas DataFrames using these column names
# Using NumPy arrays or wrong column order will cause sklearn warnings and accuracy loss
FEATURE_COLUMNS = [
    "soil_moisture",
    "soil_pH",
    "temperature",
    "rainfall",
    "humidity"
]


CROP_RULES = {
    'Rice': {
        'pH_min': 5.8, 'pH_max': 7.5,
        'temperature_min': 20, 'temperature_max': 30,
        'moisture_min': 40, 'moisture_max': 100
    },
    'Wheat': {
        'pH_min': 6.0, 'pH_max': 8.0,
        'temperature_min': 15, 'temperature_max': 25,
        'moisture_min': 25, 'moisture_max': 75
    },
    'Maize': {
        'pH_min': 5.8, 'pH_max': 7.5,
        'temperature_min': 18, 'temperature_max': 27,
        'moisture_min': 30, 'moisture_max': 85
    },
    'Soybean': {
        'pH_min': 6.0, 'pH_max': 7.8,
        'temperature_min': 18, 'temperature_max': 28,
        'moisture_min': 30, 'moisture_max': 75
    },
    'Cotton': {
        'pH_min': 6.0, 'pH_max': 8.0,
        'temperature_min': 20, 'temperature_max': 32,
        'moisture_min': 25, 'moisture_max': 70
    }
}


class PredictionService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        encoders_dir = os.path.join(os.path.dirname(__file__), 'encoders')
        
        try:
            self.yield_model = joblib.load(os.path.join(models_dir, 'yield_predictor.pkl'))
            self.irrigation_model = joblib.load(os.path.join(models_dir, 'irrigation_model.pkl'))
            
            # CRITICAL: Load fitted encoders - NEVER refit them
            # Refitting changes class mappings and destroys prediction accuracy
            # Example: Training [No=0, Yes=1] but refitting gives [Yes=0, No=1] = inverted predictions
            self.irrigation_encoder = joblib.load(os.path.join(encoders_dir, 'irrigation_encoder.pkl'))
            self.metadata = joblib.load(os.path.join(encoders_dir, 'model_metadata.pkl'))
            
            # Use global constant for feature columns to ensure consistency
            self.feature_columns = FEATURE_COLUMNS
            self._initialized = True
            
        except FileNotFoundError as e:
            raise Exception(f"Model files not found. Please train models first: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        errors = {}
        
        for feature in self.feature_columns:
            if feature not in input_data:
                errors[feature] = f"Missing required field: {feature}"
                continue
                
            try:
                value = float(input_data[feature])
                if value != value or value == float('inf') or value == float('-inf'):
                    errors[feature] = "Value must be a finite number"
            except (TypeError, ValueError):
                errors[feature] = "Value must be a number"
        
        return errors
    
    def filter_crops_by_rules(self, input_data: Dict[str, float], relax_factor: float = 0.0) -> List[str]:
        soil_moisture = input_data.get('soil_moisture', 50)
        soil_pH = input_data.get('soil_pH', 7)
        temperature = input_data.get('temperature', 25)
        
        suitable_crops = []
        
        for crop, rules in CROP_RULES.items():
            pH_min = rules['pH_min'] - (rules['pH_min'] * relax_factor)
            pH_max = rules['pH_max'] + (rules['pH_max'] * relax_factor)
            temp_min = rules['temperature_min'] - (rules['temperature_min'] * relax_factor)
            temp_max = rules['temperature_max'] + (rules['temperature_max'] * relax_factor)
            moisture_min = rules['moisture_min'] - (rules['moisture_min'] * relax_factor)
            moisture_max = rules['moisture_max'] + (rules['moisture_max'] * relax_factor)
            
            if (pH_min <= soil_pH <= pH_max and
                temp_min <= temperature <= temp_max and
                moisture_min <= soil_moisture <= moisture_max):
                suitable_crops.append(crop)
        
        return suitable_crops
    
    def rank_crops_by_yield(self, suitable_crops: List[str], input_data: Dict[str, float]) -> List[Dict[str, Any]]:
        converted_data = self.convert_input(input_data)
        
        # CRITICAL: Use pandas DataFrame with exact column names from training
        # sklearn models trained on DataFrames expect DataFrames during prediction
        # Using NumPy arrays causes feature name warnings and can reduce accuracy
        features_df = pd.DataFrame([converted_data], columns=FEATURE_COLUMNS)
        
        crop_yields = []
        
        for crop in suitable_crops:
            # Predict using DataFrame to maintain feature names
            yield_pred = float(self.yield_model.predict(features_df)[0])
            crop_yields.append({
                'crop': crop,
                'predicted_yield': yield_pred
            })
        
        crop_yields.sort(key=lambda x: x['predicted_yield'], reverse=True)
        
        for idx, item in enumerate(crop_yields):
            item['rank'] = idx + 1
        
        return crop_yields[:3]
        

    def calculate_soil_health_index(self, input_data):
        weights = {
            'soil_moisture': 0.2,
            'soil_pH': 0.3,
            'temperature': 0.15,
            'rainfall': 0.2,
            'humidity': 0.15
        }
        
        # Normalize and calculate health index
        normalized = {}
        
        # Soil moisture: optimal around 50%, scales 0-100
        normalized['soil_moisture'] = min(100, abs(50 - input_data.get('soil_moisture', 50)) / 50 * 100)
        
        # Soil pH: optimal around 7, scales 0-100
        normalized['soil_pH'] = max(0, 100 - abs(7 - input_data.get('soil_pH', 7)) * 10)
        
        # Temperature: optimal around 25°C, scales 0-100
        normalized['temperature'] = max(0, 100 - abs(25 - input_data.get('temperature', 25)) * 2)
        
        # Rainfall: normalize to 0-100
        normalized['rainfall'] = min(100, (input_data.get('rainfall', 150) / 200) * 100)
        
        # Humidity: optimal around 65%, scales 0-100
        normalized['humidity'] = max(0, 100 - abs(65 - input_data.get('humidity', 65)) * 0.8)
        
        # Calculate weighted health index
        health_index = sum(normalized.get(feature, 50) * weight for feature, weight in weights.items())
        
        return max(0, min(100, health_index))
    
    def convert_input(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        converted = {}
        for feature in self.feature_columns:
            try:
                converted[feature] = float(input_data[feature])
            except (TypeError, ValueError, KeyError):
                converted[feature] = 0.0
        return converted
    
    def predict_all(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validation_errors = self.validate_input(input_data)
        if validation_errors:
            raise ValueError(f"Invalid input: {validation_errors}")
        
        converted_data = self.convert_input(input_data)
        
        try:
            # CRITICAL: Use pandas DataFrame with exact column names from training
            # sklearn models trained on DataFrames expect DataFrames during prediction
            # Using NumPy arrays causes feature name warnings and can reduce accuracy
            features_df = pd.DataFrame([converted_data], columns=FEATURE_COLUMNS)
            
            suitable_crops = self.filter_crops_by_rules(converted_data)
            
            if not suitable_crops:
                suitable_crops = self.filter_crops_by_rules(converted_data, relax_factor=0.1)
            
            if not suitable_crops:
                suitable_crops = list(CROP_RULES.keys())
            
            ranked_crops = self.rank_crops_by_yield(suitable_crops, converted_data)
            
            top_crop = ranked_crops[0] if ranked_crops else {'crop': 'Unknown', 'predicted_yield': 0}
            
            # Predict using DataFrame to maintain feature names
            # CRITICAL: Use loaded encoder for inverse_transform - never refit
            irrigation_pred = self.irrigation_model.predict(features_df)[0]
            irrigation_required = self.irrigation_encoder.inverse_transform([irrigation_pred])[0]
            
            soil_health_index = self.calculate_soil_health_index(converted_data)
            
            return {
                'success': True,
                'crop_type': top_crop['crop'],
                'crop_confidence': 0.85,
                'yield_prediction': top_crop['predicted_yield'],
                'crop_recommendations': ranked_crops,
                'irrigation_required': irrigation_required,
                'soil_health_index': soil_health_index,
                'input_data': converted_data
            }
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")


def get_prediction_service() -> PredictionService:
    try:
        return PredictionService()
    except Exception as e:
        raise Exception(f"Failed to initialize prediction service: {str(e)}")


def predict_from_json(input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        service = get_prediction_service()
        result = service.predict_all(input_data)
        return result
    except ValueError as e:
        return {
            'success': False,
            'error': 'validation_error',
            'message': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': 'prediction_error',
            'message': str(e)
        }


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os


DATASET_PATH = r"C:\Users\jones\Downloads\clean_soil_crop_dataset.csv"
ENCODERS_DIR = os.path.join(os.path.dirname(__file__), 'encoders')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


class ModelTrainer:
    def __init__(self):
        self.df = None
        self.crop_encoder = LabelEncoder()
        self.irrigation_encoder = LabelEncoder()
        self.feature_columns = ['soil_moisture', 'soil_pH', 'temperature', 'rainfall', 'humidity']
        
    def load_data(self):
        self.df = pd.read_csv(DATASET_PATH)
        
        # Remove rare crops to improve model accuracy
        # Rare classes cause: 1) Overfitting (model memorizes few samples)
        # 2) Poor generalization 3) Class imbalance issues 4) Unreliable predictions
        print("\n=== Before Filtering ===")
        print(self.df['crop_type'].value_counts())
        print(f"Total samples: {len(self.df)}")
        
        # Keep only top 5-6 most frequent crops (data-driven selection)
        top_crops = self.df['crop_type'].value_counts().head(6).index.tolist()
        self.df = self.df[self.df['crop_type'].isin(top_crops)]
        
        print("\n=== After Filtering ===")
        print(self.df['crop_type'].value_counts())
        print(f"Total samples: {len(self.df)}")
        print(f"Kept crops: {top_crops}")
        
        available_features = [col for col in self.feature_columns if col in self.df.columns]
        if not available_features:
            all_columns = [col for col in self.df.columns if col not in ['crop_type', 'yield', 'irrigation_required']]
            self.feature_columns = all_columns[:5] if len(all_columns) >= 5 else all_columns
        else:
            self.feature_columns = available_features
        
        # Create engineered features
        self.create_engineered_features()
            
        return self.df
    
    def create_engineered_features(self):
        # Tree-based models benefit from engineered features because:
        # 1) They capture non-linear relationships explicitly
        # 2) Reduce search space for optimal splits
        # 3) Provide domain knowledge shortcuts
        # 4) Improve interpretability of feature importance
        
        # moisture_temp_ratio: High values = cool + moist (good for crops)
        self.df['moisture_temp_ratio'] = self.df['soil_moisture'] / (self.df['temperature'] + 1e-6)
        
        # rainfall_humidity_index: Combined water availability indicator
        self.df['rainfall_humidity_index'] = self.df['rainfall'] * self.df['humidity']
        
        # stress_index: High values = hot + dry (crop stress)
        self.df['stress_index'] = self.df['temperature'] / (self.df['soil_moisture'] + 1e-6)
        
        # Update feature columns to include engineered features
        self.feature_columns = self.feature_columns + [
            'moisture_temp_ratio',
            'rainfall_humidity_index',
            'stress_index'
        ]
        
        print(f"\nEngineered features created: {self.feature_columns[-3:]}")
    
    def calculate_soil_health_index(self, row):
        weights = {
            'soil_moisture': 0.2,
            'soil_pH': 0.3,
            'temperature': 0.15,
            'rainfall': 0.2,
            'humidity': 0.15
        }
        
        normalized = {}
        for feature in self.feature_columns:
            if feature in row:
                min_val = self.df[feature].min()
                max_val = self.df[feature].max()
                normalized[feature] = (row[feature] - min_val) / (max_val - min_val) * 100
        
        if 'soil_pH' in normalized:
            optimal_ph = 6.5
            ph_deviation = abs(row['soil_pH'] - optimal_ph)
            normalized['soil_pH'] = max(0, 100 - (ph_deviation * 20))
        
        score = 0
        for feature in self.feature_columns:
            if feature in normalized and feature in weights:
                score += normalized[feature] * weights[feature]
            elif feature in normalized:
                score += normalized[feature] * (1.0 / len(self.feature_columns))
                
        return min(100, max(0, score))
    
    def prepare_soil_health_data(self):
        self.df['soil_health_index'] = self.df.apply(self.calculate_soil_health_index, axis=1)
        return self.df
    
    def train_crop_classifier(self):
        X = self.df[self.feature_columns]
        
        # CRITICAL: Fit encoder ONCE during training only
        # Encoder mismatch destroys accuracy because:
        # 1) Different class order means model predicts wrong labels
        # 2) New classes during prediction cause errors
        # 3) Missing classes break inverse_transform
        # Example: Training [Rice=0, Wheat=1] but prediction [Wheat=0, Rice=1] = 100% wrong
        y = self.crop_encoder.fit_transform(self.df['crop_type'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # RandomForest improves generalization by:
        # 1) Averaging predictions from multiple trees reduces overfitting
        # 2) Bootstrap sampling creates diverse trees
        # 3) Random feature selection at each split reduces correlation between trees
        # 4) More stable and robust than single decision tree
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== Crop Suitability Model (Random Forest) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classes: {self.crop_encoder.classes_.tolist()}")
        
        return model, accuracy
    
    def train_yield_predictor(self):
        if 'yield' not in self.df.columns:
            base_yields = {crop: np.random.uniform(20, 100) for crop in self.df['crop_type'].unique()}
            self.df['yield'] = self.df.apply(
                lambda row: base_yields[row['crop_type']] * (1 + np.random.uniform(-0.3, 0.3)),
                axis=1
            )
        
        X = self.df[self.feature_columns]
        y = self.df['yield']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== Yield Prediction Model (Random Forest) ===")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return model, r2
    
    def train_irrigation_model(self):
        if 'irrigation_required' not in self.df.columns:
            self.df['irrigation_required'] = self.df.apply(
                lambda row: 'Yes' if row['rainfall'] < self.df['rainfall'].median() else 'No',
                axis=1
            )
        
        X = self.df[self.feature_columns]
        
        # CRITICAL: Fit encoder ONCE during training only
        # Never refit during prediction - use saved encoder
        y = self.irrigation_encoder.fit_transform(self.df['irrigation_required'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rule_based = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        rule_based.fit(X_train, y_train)
        
        y_pred = rule_based.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== Irrigation Requirement Model (Rule-based) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classes: {self.irrigation_encoder.classes_.tolist()}")
        
        return rule_based, accuracy
    
    def save_models(self, crop_model, yield_model, irrigation_model):
        os.makedirs(ENCODERS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        joblib.dump(crop_model, os.path.join(MODELS_DIR, 'crop_classifier.pkl'))
        joblib.dump(yield_model, os.path.join(MODELS_DIR, 'yield_predictor.pkl'))
        joblib.dump(irrigation_model, os.path.join(MODELS_DIR, 'irrigation_model.pkl'))
        
        # CRITICAL: Save fitted encoders - NEVER refit during prediction
        # Encoders must be loaded and used as-is to maintain label consistency
        joblib.dump(self.crop_encoder, os.path.join(ENCODERS_DIR, 'crop_encoder.pkl'))
        joblib.dump(self.irrigation_encoder, os.path.join(ENCODERS_DIR, 'irrigation_encoder.pkl'))
        
        metadata = {
            'feature_columns': self.feature_columns,
            'crop_classes': self.crop_encoder.classes_.tolist(),
            'irrigation_classes': self.irrigation_encoder.classes_.tolist(),
        }
        joblib.dump(metadata, os.path.join(ENCODERS_DIR, 'model_metadata.pkl'))
        
        print(f"\n=== Models Saved ===")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Encoders directory: {ENCODERS_DIR}")
        print(f"Crop encoder classes: {self.crop_encoder.classes_.tolist()}")
        print(f"Irrigation encoder classes: {self.irrigation_encoder.classes_.tolist()}")
    
    def train_all_models(self):
        print("Loading dataset...")
        self.load_data()
        print(f"Features: {self.feature_columns}")
        
        print("\nPreparing soil health index...")
        self.prepare_soil_health_data()
        
        print("\nTraining models...")
        crop_model, crop_acc = self.train_crop_classifier()
        yield_model, yield_r2 = self.train_yield_predictor()
        irrigation_model, irr_acc = self.train_irrigation_model()
        
        self.save_models(crop_model, yield_model, irrigation_model)
        
        print("\n=== Training Complete ===")
        print(f"Crop Classifier Accuracy: {crop_acc:.4f}")
        print(f"Yield Predictor R2: {yield_r2:.4f}")
        print(f"Irrigation Model Accuracy: {irr_acc:.4f}")


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_all_models()

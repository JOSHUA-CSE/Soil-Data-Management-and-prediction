import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os


DATASET_PATH = r"C:\Users\jones\Downloads\clean_soil_crop_dataset.csv"
ENCODERS_DIR = os.path.join(os.path.dirname(__file__), 'encoders')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    
    label_encoder = LabelEncoder()
    df['crop_type_encoded'] = label_encoder.fit_transform(df['crop_type'])
    
    feature_columns = [col for col in df.columns if col not in ['crop_type', 'crop_type_encoded']]
    X = df[feature_columns]
    y = df['crop_type_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs(ENCODERS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    encoder_path = os.path.join(ENCODERS_DIR, 'crop_type_encoder.pkl')
    joblib.dump(label_encoder, encoder_path)
    
    metadata = {
        'feature_columns': feature_columns,
        'classes': label_encoder.classes_.tolist(),
        'n_samples': len(df),
        'n_features': len(feature_columns),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    metadata_path = os.path.join(ENCODERS_DIR, 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    
    return X_train, X_test, y_train, y_test, label_encoder, feature_columns


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, encoder, features = load_and_preprocess_data()
    
    print(f"Dataset loaded successfully")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {features}")
    print(f"Classes: {encoder.classes_.tolist()}")
    print(f"Encoder saved to: encoders/crop_type_encoder.pkl")

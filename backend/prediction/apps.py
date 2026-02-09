from django.apps import AppConfig


class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction'
    
    def ready(self):
        try:
            from .prediction_service import get_prediction_service
            service = get_prediction_service()
            print(f"✓ ML models loaded successfully")
            print(f"✓ Feature columns: {service.feature_columns}")
        except Exception as e:
            print(f"✗ Warning: Could not load ML models: {str(e)}")
            print(f"✗ Please run train_models.py to train models first")

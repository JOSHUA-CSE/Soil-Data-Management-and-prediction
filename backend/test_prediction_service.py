import os
import sys
import django

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from prediction.prediction_service import predict_from_json


def test_prediction_service():
    print("=== Testing Prediction Service ===\n")
    
    test_cases = [
        {
            'name': 'Valid input',
            'data': {
                'soil_moisture': 65.5,
                'soil_pH': 6.8,
                'temperature': 25.3,
                'rainfall': 150.2,
                'humidity': 75.8
            }
        },
        {
            'name': 'String numbers',
            'data': {
                'soil_moisture': '70',
                'soil_pH': '6.5',
                'temperature': '22',
                'rainfall': '120',
                'humidity': '80'
            }
        },
        {
            'name': 'Missing field',
            'data': {
                'soil_moisture': 65.5,
                'soil_pH': 6.8,
                'temperature': 25.3,
                'humidity': 75.8
            }
        },
        {
            'name': 'Invalid value',
            'data': {
                'soil_moisture': 65.5,
                'soil_pH': 'invalid',
                'temperature': 25.3,
                'rainfall': 150.2,
                'humidity': 75.8
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['data']}")
        
        result = predict_from_json(test_case['data'])
        
        if result.get('success'):
            print(f"✓ Success")
            print(f"  Crop: {result['crop_type']} (confidence: {result['crop_confidence']:.2%})")
            print(f"  Yield: {result['yield_prediction']:.2f}")
            print(f"  Irrigation: {result['irrigation_required']}")
            print(f"  Soil Health: {result['soil_health_index']:.2f}/100")
        else:
            print(f"✗ Failed: {result.get('error')}")
            print(f"  Message: {result.get('message')}")
        
        print()


if __name__ == '__main__':
    test_prediction_service()

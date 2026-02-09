import os
import sys
import django

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from pymongo import MongoClient
from django.conf import settings


def test_mongodb_connection():
    try:
        mongo_settings = settings.MONGODB_SETTINGS
        client = MongoClient(
            host=mongo_settings['host'],
            port=mongo_settings['port'],
            serverSelectionTimeoutMS=5000
        )
        
        client.server_info()
        print("✓ MongoDB connection successful")
        
        db = client[mongo_settings['database']]
        print(f"✓ Database '{mongo_settings['database']}' accessible")
        
        collection = db[mongo_settings['collection']]
        print(f"✓ Collection '{mongo_settings['collection']}' accessible")
        
        test_doc = {
            'nitrogen': 90.0,
            'phosphorus': 42.0,
            'potassium': 43.0,
            'ph_value': 6.5,
            'rainfall': 202.9,
            'temperature': 25.5,
            'humidity': 80.5,
            'predicted_crop': 'rice',
            'confidence': 0.95,
            'timestamp': '2026-02-08T00:00:00Z'
        }
        
        result = collection.insert_one(test_doc)
        print(f"✓ Test document inserted with ID: {result.inserted_id}")
        
        count = collection.count_documents({})
        print(f"✓ Total documents in collection: {count}")
        
        collection.delete_one({'_id': result.inserted_id})
        print("✓ Test document cleaned up")
        
        client.close()
        print("\n✓ All MongoDB tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ MongoDB connection failed: {str(e)}")
        return False


if __name__ == '__main__':
    test_mongodb_connection()

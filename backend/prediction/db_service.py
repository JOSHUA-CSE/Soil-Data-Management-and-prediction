from pymongo import MongoClient
from django.conf import settings
from datetime import datetime


class MongoDBService:
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        mongo_settings = settings.MONGODB_SETTINGS
        self._client = MongoClient(
            host=mongo_settings['host'],
            port=mongo_settings['port']
        )
        self._db = self._client[mongo_settings['database']]
    
    @property
    def db(self):
        return self._db
    
    @property
    def prediction_logs(self):
        return self._db[settings.MONGODB_SETTINGS['collection']]
    
    def save_prediction(self, input_data, prediction_result):
        document = {
            'nitrogen': input_data.get('nitrogen'),
            'phosphorus': input_data.get('phosphorus'),
            'potassium': input_data.get('potassium'),
            'ph_value': input_data.get('ph_value'),
            'rainfall': input_data.get('rainfall'),
            'temperature': input_data.get('temperature'),
            'humidity': input_data.get('humidity'),
            'predicted_crop': prediction_result.get('crop'),
            'confidence': prediction_result.get('confidence'),
            'timestamp': datetime.utcnow()
        }
        result = self.prediction_logs.insert_one(document)
        return str(result.inserted_id)
    
    def get_all_predictions(self, limit=100):
        return list(self.prediction_logs.find().sort('timestamp', -1).limit(limit))
    
    def get_prediction_by_id(self, prediction_id):
        from bson import ObjectId
        return self.prediction_logs.find_one({'_id': ObjectId(prediction_id)})
    
    def close(self):
        if self._client:
            self._client.close()


mongo_service = MongoDBService()

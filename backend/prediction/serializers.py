from rest_framework import serializers
from .models import PredictionLog


class PredictionInputSerializer(serializers.Serializer):
    soil_moisture = serializers.FloatField(required=True, min_value=0, max_value=100)
    soil_pH = serializers.FloatField(required=True, min_value=0, max_value=14)
    temperature = serializers.FloatField(required=True, min_value=-50, max_value=60)
    rainfall = serializers.FloatField(required=True, min_value=0, max_value=5000)
    humidity = serializers.FloatField(required=True, min_value=0, max_value=100)


class PredictionOutputSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    crop_type = serializers.CharField(required=False)
    crop_confidence = serializers.FloatField(required=False)
    yield_prediction = serializers.FloatField(required=False)
    irrigation_required = serializers.CharField(required=False)
    soil_health_index = serializers.FloatField(required=False)
    input_data = serializers.DictField(required=False)
    prediction_id = serializers.CharField(required=False)
    timestamp = serializers.DateTimeField(required=False)
    error = serializers.CharField(required=False)
    message = serializers.CharField(required=False)


class PredictionHistorySerializer(serializers.Serializer):
    id = serializers.CharField()
    soil_moisture = serializers.FloatField()
    soil_pH = serializers.FloatField()
    temperature = serializers.FloatField()
    rainfall = serializers.FloatField()
    humidity = serializers.FloatField()
    crop_type = serializers.CharField()
    crop_confidence = serializers.FloatField()
    yield_prediction = serializers.FloatField()
    irrigation_required = serializers.CharField()
    soil_health_index = serializers.FloatField()
    timestamp = serializers.DateTimeField()

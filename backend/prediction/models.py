from django.db import models


class PredictionLog(models.Model):
    soil_moisture = models.FloatField()
    soil_pH = models.FloatField()
    temperature = models.FloatField()
    rainfall = models.FloatField()
    humidity = models.FloatField()
    crop = models.CharField(max_length=100)
    yield_value = models.FloatField()
    irrigation = models.CharField(max_length=100)
    soil_health = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'prediction_logs'
        ordering = ['-created_at']

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionInputSerializer, PredictionOutputSerializer, PredictionHistorySerializer
from .prediction_service import predict_from_json
from .db_service import mongo_service
from datetime import datetime
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)


class PredictView(APIView):
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        
        if not serializer.is_valid():
            logger.error(f"Validation errors: {serializer.errors}")
            return Response({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid input data',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        input_data = serializer.validated_data
        
        try:
            prediction_result = predict_from_json(input_data)
            
            if not prediction_result.get('success'):
                logger.warning(f"Prediction failed: {prediction_result}")
                return Response(prediction_result, status=status.HTTP_400_BAD_REQUEST)
            
            mongo_doc = {
                'soil_moisture': input_data['soil_moisture'],
                'soil_pH': input_data['soil_pH'],
                'temperature': input_data['temperature'],
                'rainfall': input_data['rainfall'],
                'humidity': input_data['humidity'],
                'crop_type': prediction_result['crop_type'],
                'crop_recommendations': prediction_result.get('crop_recommendations', []),
                'crop_confidence': prediction_result['crop_confidence'],
                'yield_prediction': prediction_result['yield_prediction'],
                'irrigation_required': prediction_result['irrigation_required'],
                'soil_health_index': prediction_result['soil_health_index'],
                'timestamp': datetime.utcnow()
            }
            
            result = mongo_service.prediction_logs.insert_one(mongo_doc)
            prediction_id = str(result.inserted_id)
            
            response_data = {
                **prediction_result,
                'prediction_id': prediction_id,
                'timestamp': mongo_doc['timestamp'].isoformat()
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            return Response({
                'success': False,
                'error': 'server_error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class HistoryView(APIView):
    def get(self, request):
        try:
            limit = int(request.query_params.get('limit', 100))
            limit = min(limit, 1000)
            
            predictions = list(
                mongo_service.prediction_logs
                .find()
                .sort('timestamp', -1)
                .limit(limit)
            )
            
            for pred in predictions:
                pred['id'] = str(pred['_id'])
                del pred['_id']
            
            return Response({
                'success': True,
                'count': len(predictions),
                'predictions': predictions
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'success': False,
                'error': 'server_error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictionDetailView(APIView):
    def get(self, request, prediction_id):
        try:
            prediction = mongo_service.prediction_logs.find_one({'_id': ObjectId(prediction_id)})
            
            if not prediction:
                return Response({
                    'success': False,
                    'error': 'not_found',
                    'message': 'Prediction not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Transform crop_recommendations to match frontend expectations
            recommendations = []
            for rec in prediction.get('crop_recommendations', []):
                recommendations.append({
                    'crop': rec['crop'],
                    'yield': rec['predicted_yield'],
                    'rank': rec.get('rank', 0)
                })
            
            # Transform MongoDB document to match frontend expectations
            response_data = {
                'id': str(prediction['_id']),
                'soil_moisture': prediction['soil_moisture'],
                'soil_pH': prediction['soil_pH'],
                'temperature': prediction['temperature'],
                'rainfall': prediction['rainfall'],
                'humidity': prediction['humidity'],
                'crop_type': prediction['crop_type'],
                'recommendations': recommendations,
                'irrigation': 'Yes' if prediction['irrigation_required'] else 'No',
                'explanation': f"Based on soil conditions and climate factors, {prediction['crop_type']} is the most suitable crop with an expected yield of {prediction['yield_prediction']:.2f} kg/ha.",
                'yield_prediction': prediction['yield_prediction'],
                'soil_health_index': prediction['soil_health_index'],
                'timestamp': prediction['timestamp'].isoformat()
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except ValueError:
            return Response({
                'success': False,
                'error': 'invalid_id',
                'message': 'Invalid prediction ID format'
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception(f"Error fetching prediction detail: {str(e)}")
            return Response({
                'success': False,
                'error': 'server_error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

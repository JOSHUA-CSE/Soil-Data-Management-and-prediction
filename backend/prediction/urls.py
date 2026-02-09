from django.urls import path
from .views import PredictView, HistoryView, PredictionDetailView

urlpatterns = [
    path('predict', PredictView.as_view(), name='predict'),
    path('history', HistoryView.as_view(), name='history'),
    path('prediction/<str:prediction_id>', PredictionDetailView.as_view(), name='prediction_detail'),
]

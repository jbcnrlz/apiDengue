from django.urls import path
from . import views

urlpatterns = [
    path('treino/', views.trainModel, name='train-model'),
    path('teste/', views.doPrediction, name='prediction-model'),
]
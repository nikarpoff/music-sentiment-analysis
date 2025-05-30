import os

from django.shortcuts import render
from rest_framework.response import Response 
from rest_framework.views import APIView

from server_app.predict import Prediction


class MoodPrediction(APIView): 
    """
    Prediction of a mood of the user song.
    """
    def post(self,request):
        # Make prediction.
        predictor = Prediction()
        response_dict = predictor.predict_by_file(request)

        # Form response and send it.
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status_value)

class MoodPredictionByLink(APIView): 
    """
    Prediction of a mood of the user link to song.
    """
    def post(self,request):
        # Make prediction.
        predictor = Prediction()
        response_dict = predictor.predict_with_link(request)

        # Form response and send it.
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status_value)
    
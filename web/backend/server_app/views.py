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
        response_dict = predictor.predict(request)

        # Form response and send it.
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status_value)
    
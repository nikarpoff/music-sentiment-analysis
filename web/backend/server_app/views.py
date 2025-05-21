import os

from django.shortcuts import render
from rest_framework.response import Response 
from rest_framework.views import APIView
from django.views.generic import TemplateView, View
from django.http import FileResponse, HttpResponseNotFound

from server_app.predict import Prediction
from backend.settings import BASE_DIR


class HomeTemplateView(TemplateView):
    """
    Home page endpoint.
    """
    template_name = 'index.html'


class ManifestView(View):
    def get(self, request):
        file_path = os.path.join(BASE_DIR, 'build/manifest.json')
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), content_type='application/json')
        return HttpResponseNotFound()


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
    
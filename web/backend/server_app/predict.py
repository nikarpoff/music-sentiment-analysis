import os
import re
import requests
from rest_framework import status
from dotenv import load_dotenv

load_dotenv()

def extract_track_id(url):
    pattern = r'jamendo\.com/track/(\d+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

class Prediction:
    def __init__(self):
        self.client_id = os.getenv('JAMENDO_CLIENT_ID')

    def _form_ok_response(self, response):
        result_dict = {}
        return_dict = {}

        data = response.json()

        # Form result dictionary and send response.
        probs_dict = {}
        classes = data['classes']
        probs = data['probs']

        for target_class, prob in zip(classes, probs):
            probs_dict[target_class] = prob

        result_dict['predict'] = data['predict']
        result_dict['probs'] = probs_dict

        return_dict['response'] = result_dict
        return_dict['status'] = status.HTTP_200_OK
        return return_dict

    def _form_not_ok_response(self, text, status_code):
        return_dict = {}
        message = f"Exception while prediction: {text}"
        return_dict['response'] = text
        return_dict['status'] = status_code
        print(message)
        return return_dict
    
    def _form_bad_request(self, text):
        return self._form_not_ok_response(text, status.HTTP_400_BAD_REQUEST)
    
    def _form_internal_server_error(self, text):
        return self._form_not_ok_response(text, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _form_forbidden(self, text):
        return self._form_not_ok_response(text, status.HTTP_403_FORBIDDEN)
    
    def _form_not_found(self, text):
        return self._form_not_ok_response(text, status.HTTP_404_NOT_FOUND)

    def predict_by_file(self, request):
        # In this request file required!
        if 'audio' not in request.FILES:
            return self._form_bad_request("Аудио-файл не был найден в запросе!")

        audio_file = request.FILES['audio']

        # Send audio file to the model for prediction.
        try:
            response = requests.post(
                'http://localhost:5000/api/model/predict',
                files={'file': (audio_file.name, audio_file, audio_file.content_type)}
            )
        except Exception as e:
            print("Error while send prediction to model: ", e)
            return self._form_internal_server_error("Модель недоступна. Попробуйте позже!")

        if response.status_code != 200:
            print("Model prediction failed with status code: ", response.status_code)
            return self._form_internal_server_error("Модель недоступна. Попробуйте позже!")
            
        return self._form_ok_response(response)

    def predict_with_link(self, request):
        audio_url = request.data.get('link')
        if not audio_url:
            return self._form_bad_request("Ссылка на аудио не была найдена в запросе!")

        # Get track_id from link
        track_id = extract_track_id(audio_url)

        if track_id is None or not track_id.isdigit():
            return self._form_bad_request(f"В ссылке не был найден номер трека. Ожидаемая ссылка: 'https://www.jamendo.com/track/<track_id>/...'; передано: {audio_url}")

        if not self.client_id:
            print("Client ID not found in environment!")
            return self._form_internal_server_error("На сервере произошла временная ошибка. Попробуйте позже!")

        # Get track info
        track_info_url = f'https://api.jamendo.com/v3.0/tracks?client_id={self.client_id}&id={track_id}'
        try:
            response = requests.get(track_info_url)
        except Exception as e:
            print("Error while send prediction to Jamendo: ", e)
            return self._form_internal_server_error("Сервера Jamendo недоступны, попробуйте позже!")

        data = response.json()

        if not data['results']:
            return self._form_not_found(f"Трек {track_id} не был найден.")

        # Check if there download available
        track = data['results'][0]
        download_url = track.get('audiodownload')
        if not track.get('audiodownload_allowed') or not download_url:
            return self._form_forbidden(f"Загрузка этого трека не разрешена автором. Простите за неудобства.")

        # Try to load file
        try:
            audio_response = requests.get(download_url, stream=True)
        except Exception as e:
            print("Error while send prediction to Jamendo: ", e)
            return self._form_internal_server_error("Сервера Jamendo недоступны, попробуйте позже!")
        
        if audio_response.status_code != 200:
            return self._form_internal_server_error("При загрузке файла на сервере произошла ошибка! Попробуйте позже!")

        # Send audio file to the model.
        try:
            response = requests.post(
                'http://localhost:5000/api/model/predict',
                files={'file': ('audio.mp3', audio_response.raw, 'audio/mpeg')}
            )
        except Exception as e:
            print("Error while send prediction to model: ", e)
            return self._form_internal_server_error("Модель недоступна. Попробуйте позже!")
        
        if response.status_code != 200:
            print("Model prediction failed with status code: ", response.status_code)
            return self._form_internal_server_error("Модель недоступна. Попробуйте позже!")

        return self._form_ok_response(response)


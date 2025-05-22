import requests
from rest_framework import status

class Prediction:
    def predict(self, request):
        return_dict = {}
        result_dict = {}

        try:
            # In this request file required!
            if 'audio' not in request.FILES:
                raise ValueError("No audio part in the request.")

            audio_file = request.FILES['audio']

            # Send audio file to the model for prediction.
            response = requests.post(
                'http://localhost:5000/api/model/predict',
                files={'file': (audio_file.name, audio_file, audio_file.content_type)}
            )

            if response.status_code != 200:
                raise ValueError("Model prediction failed: " + str(data))
            
            data = response.json()

            # Form result dictionary and send response.
            result_dict['prediction'] = data['predict']
            result_dict['logits'] = data['logits']
            result_dict['classes'] = data['classes']

            return_dict['response'] = result_dict
            return_dict['status'] = status.HTTP_200_OK
            return return_dict

        except Exception as e:
            return_dict['response'] = "Exception when prediction: " + str(e)
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict

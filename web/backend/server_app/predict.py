import os 
import pandas as pd 
from rest_framework import status 
import json

base_path = os.getcwd()
models_path = os.path.join(base_path, 'data')

class Prediction:
    def predict(self,request):
        return_dict = dict()

        try:
            request_body = request.body 
            decoded_request_body = request_body.decode('utf8').replace("'",'"')
            request_dict=json.loads(decoded_request_body)

            print(request_dict)

            df_pred=pd.json_normalize(request_dict)

            # pickle_file = os.path.normpath(pickle_path+os.sep+'model.sav')
            # model=pickle.load(open(pickle_file,'rb'))
            # prediction=model.predict(df_pred)
            # print(prediction)

            print(df_pred)

            request_dict['prediction'] = df_pred 
            return_dict['response'] = request_dict
            return_dict['status'] = status.HTTP_200_OK

            return return_dict

        except Exception as e: 
            return_dict['response'] = "Exception when prediction: " + str(e.__str__)
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict

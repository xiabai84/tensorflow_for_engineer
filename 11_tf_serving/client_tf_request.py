import requests
import json

image = r"./test2.jpg"
URL = "http://localhost:8501/v1/models/my_model:predict"
headers = {"content-type": "application/json"}
data = [1.8]
print(data)
body = {
    "signature_name": "prediction",
    "instances": [{"input": data}]
    }
r = requests.post(URL, data=json.dumps(body), headers = headers)
print(r.text)


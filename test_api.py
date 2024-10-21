import json
import requests
url="http://localhost:4000/prediction_api"
with open("notebooks/api_test_data.json", 'r') as file:
    data = json.load(file)
    r = requests.post(url, json=data)
print(r.text)
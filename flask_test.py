import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('../datasets/test_animals/cat/cats_00001.jpg','rb')})

print(resp.json())
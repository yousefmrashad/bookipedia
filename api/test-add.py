import requests

url = "http://localhost:8000/add_document/5"
params = {
    "url": "https://arxiv.org/pdf/2309.12871",
    "lib_doc": "false"
}

response = requests.post(url, params=params)
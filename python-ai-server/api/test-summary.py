import requests

url = "http://localhost:8000/text_summary"

params = {
    "doc_id":"1",
    "pages":[2,3]
}

# Print response content
response = requests.get(url, params=params, stream=True)
for line in response.iter_lines():
    print(line.decode('utf-8'))
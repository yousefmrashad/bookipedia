import requests

url = "http://localhost:8000/summarize_pages/1"

params = {
    "pages":[2,3]
}

# Print response content
response = requests.get(url, params=params, stream=True)
for line in response.iter_lines():
    print(line.decode('utf-8'))
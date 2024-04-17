import requests

url = "http://localhost:8000/stream_response_and_summary"

# Parameters
params = {
    "user_prompt": "what are Tesla's major contributions?",
    "chat_summary": "Nikola Tesla was a Serbian-American inventor, electrical engineer, and futurist. He is known for his contributions.",
    "chat": """
    User: Who is Nikola Tesla?
    Answer: Nikola Tesla was a famous inventor.
    User: What is his nationality?
    Answer: He was a Serbian-American.
    """,
    "enable_web_retrieval": True
}

# Make GET request
response = requests.get(url, params=params)

# Print response content
for line in response.iter_lines():
    print(line.decode('utf-8'))
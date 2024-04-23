from root_config import *
from utils.init import *
# ================================================== #
from typing import Annotated
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from RAG.rag_pipeline import RAGPipeline
from preprocessing.document_class import Document
from preprocessing.embeddings_class import MXBAIEmbedding
from utils.db_config import DB
from piper import PiperVoice
import requests
import json
# ================================================== #

#Initializations
voice = PiperVoice.load('/home/yousef/bookipedia/python-ai-server/test-piper/en_US-amy-medium.onnx',
                        '/home/yousef/bookipedia/python-ai-server/test-piper/en_US-amy-medium.onnx.json',
                        use_cuda=False)

app = FastAPI()
embedding_model=MXBAIEmbedding()
client = DB().connect()
rag_pipeline = RAGPipeline(embedding_model, client)
background_tasks = BackgroundTasks()
# -------------------------------------------------- #

# Background Tasks
def process_document(doc: Document, is_text_based: bool = True):
    doc.preprocess(client, embedding_model)
    if is_text_based:
        # Delete the file named doc_id
        requests.post(ACKNOWLEDGE_URL, data = {"doc_id": doc.doc_id, "messge": "Document preprocessing completed."})
    else:
        with open(doc.doc_path, 'rb') as file:
            response = requests.post(POST_HOCR_URL, data = file, stream = True)
        if response.status_code == 200:
            requests.post(ACKNOWLEDGE_URL, data = {"doc_id": doc.doc_id, "messge": "Document OCR and preprocessing completed."})
        else:
            print("Failed to post HOCR file. Status code:", response.status_code)
    os.remove(doc.doc_path)

def chat_summary(chat_id:str, response: str, user_prompt: str, prev_summary:str):
    summary = rag_pipeline.generate_chat_summary(response, user_prompt, prev_summary)
    requests.post(CHAT_SUMMARY_URL, data = {"chat_id":chat_id, "summary": summary})
# -------------------------------------------------- #

# Endpoints
@app.get("/")
async def root():
    return {"message": "bookipedia"}

@app.post("/add_document")
async def add_document(doc_id: str, url: str):
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the contents of the response to it
        doc_path = doc_id + '.pdf'
        with open(doc_path, 'wb') as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)
        return {"message": "Failed to download file. Status code:", "status code": response.status_code}

    # Create a Document object and add it to the background tasks    
    doc = Document(doc_path, doc_id)
    is_text_based = doc.is_text_based_document()
    background_tasks.add_task(process_document(doc, is_text_based))
    if(is_text_based):
        return {"message": "Document is text-based. Preprocessing started."}
    else:
        return {"message": "Document is not text-based. Applying OCR."}

@app.get("/chat_response")
async def chat_response(chat_id:str,
                        user_prompt: str,
                        chat_summary: str,
                        chat: str,
                        doc_ids: Annotated[list[str] | None, Query()] = None,
                        enable_web_retrieval:bool = True):
    # Initialize RAG pipeline
    async def stream_generator():
        response = ""
        # Yield data stream
        async for chunk in rag_pipeline.generate_answer(user_prompt, chat_summary, chat, doc_ids, enable_web_retrieval):
            response += chunk
            yield chunk.encode('utf-8')
        # Yield metadata as first part of the stream
        yield b'\n\nSources: '
        yield json.dumps(rag_pipeline.metadata).encode('utf-8') + b'\n'
        # Add chat summary to background tasks
        background_tasks.add_task(chat_summary(chat_id, response, user_prompt, chat_summary))
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/tts/")
async def text_to_speech(text: str, speed: float = 1):
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        lines = text.split('\n')
        for line in lines:
            audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
            for audio_bytes in audio_stream:
                yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/x-wav")

@app.get("/tts_pages/")
async def pages_to_speech(doc_id: str, pages: Annotated[list[int] | None, Query()], speed: float = 1):
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        for page in pages:
            text = rag_pipeline.get_page_text(doc_id, page)
            lines = text.split('\n')
            for line in lines:
                audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
                for audio_bytes in audio_stream:
                    yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/x-wav")


@app.get("/summarize_pages")
async def summarize_pages(doc_id: str, pages: Annotated[list[int] | None, Query()]):
    async def stream_generator():
        # Yield data stream
        async for chunk in await rag_pipeline.summarize_pages(doc_id, pages):
            yield chunk.encode('utf-8')
            
    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
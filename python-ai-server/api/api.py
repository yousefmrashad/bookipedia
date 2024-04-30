from root_config import *
from utils.init import *
# ================================================== #
from typing import Annotated
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from bodies import *
from RAG.rag_pipeline import RAGPipeline
from preprocessing.document_class import Document
from preprocessing.embeddings_class import HFEmbedding
from utils.db_config import DB
from piper import PiperVoice
import requests
import json
# ================================================== #

#Initializations
voice = PiperVoice.load(PIPER_MODEL_PATH,
                        PIPER_CONFIG_PATH,
                        use_cuda=False)

app = FastAPI(
    title="Bookipedia AI Server",
    description="Bookipedia AI Server is an AI inference server for the Bookipedia application, which serves as an online library with a AI-powered reading assistant. The server utilizes state-of-the-art language models (LLMs), optical character recognition (OCR), and text-to-speech (TTS) features.",
    version="0.0.3"
)
embedding_model=HFEmbedding()
client = DB().connect()
rag_pipeline = RAGPipeline(embedding_model, client)
background_tasks = BackgroundTasks()
# -------------------------------------------------- #

# Background Tasks
def process_document(doc: Document, doc_id: str):
    if doc.text_based:
        doc.process_document(client, embedding_model)
        requests.post(ACKNOWLEDGE_URL + doc.doc_id, json = {"messge": "Document preprocessing completed."})
    else:
        doc.get_text_based_document()
        with open(doc.doc_path, 'rb') as file:
            response = requests.post(POST_HOCR_URL + doc_id, files = {'file':(file.name, file, 'application/pdf')})
        if response.status_code == 202:
            doc.doc_id = response.json()['file_id']
            doc.process_document(embedding_model, client) 
            requests.post(ACKNOWLEDGE_URL + doc.doc_id, json = {"messge": "Document OCR and preprocessing completed."})
            print("HOCR file posted successfully.")
        else:
            print(f"Failed to post HOCR file. Status code:{response.status_code} Text:{response.text}")
    os.remove(doc.doc_path)

def summarize_chat(chat_id:str, response: str, user_prompt: str, prev_summary:str):
    chat_summary = rag_pipeline.generate_chat_summary(response, user_prompt, prev_summary)
    response = requests.patch(CHAT_SUMMARY_URL + chat_id, json = {"chat_summary": chat_summary})
    if response.status_code == 202:
        print("Chat summary updated successfully.")
    else:
        print("Failed to update chat summary. Status code:", response.status_code)
# -------------------------------------------------- #

# Endpoints
@app.get("/")
async def root():
    return {"message": "bookipedia"}

@app.post("/add_document/{doc_id}")
async def add_document(doc_id: str, url: str, lib_doc: bool = False):
    # Send a GET request to the URL
    response = requests.get(url, stream = True)
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
    doc = Document(doc_path, doc_id, lib_doc)
    background_tasks.add_task(process_document(doc, doc_id))
    if(doc.text_based):
        return {"message": "Document is text-based. Preprocessing started.", "OCR": False}
    else:
        return {"message": "Document is not text-based. Applying OCR.", "OCR": True}

@app.get("/chat_response/{chat_id}")
async def chat_response(chat_id:str,
                        chat_params: ChatParams,
                        enable_web_retrieval:bool = False):
    # Extract parameters
    user_prompt = chat_params.user_prompt
    chat_summary = chat_params.chat_summary
    chat = chat_params.chat
    doc_ids = chat_params.doc_ids

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
        background_tasks.add_task(summarize_chat(chat_id, response, user_prompt, chat_summary))
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/tts/")
async def text_to_speech(tts_text: TTSText, speed: float = 1):
    text = tts_text.text
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        lines = text.split('\n')
        for line in lines:
            audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
            for audio_bytes in audio_stream:
                yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/raw")

@app.get("/tts_pages/{doc_id}")
async def pages_to_speech(doc_id: str, pages: Annotated[list[int], Query()], speed: float = 1):
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        for page in pages:
            text = rag_pipeline.get_page_text(doc_id, page)
            lines = text.split('\n')
            for line in lines:
                audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
                for audio_bytes in audio_stream:
                    yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/raw")


@app.get("/summarize_pages/{doc_id}")
async def summarize_pages(doc_id: str, pages: Annotated[list[int], Query()]):
    async def stream_generator():
        # Yield data stream
        async for chunk in await rag_pipeline.summarize_pages(doc_id, pages):
            yield chunk.encode('utf-8')
            
    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
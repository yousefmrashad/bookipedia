from root_config import *
from utils.init import *
# ================================================== #
from typing import Annotated
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from RAG.rag_pipeline import RAGPipeline
from preprocessing.embeddings_class import MXBAIEmbedding
from utils.db_config import DB
from piper import PiperVoice
import json

voice = PiperVoice.load('/home/yousef/bookipedia/python-ai-server/test-piper/en_US-amy-medium.onnx',
                        '/home/yousef/bookipedia/python-ai-server/test-piper/en_US-amy-medium.onnx.json',
                        use_cuda=False)

app = FastAPI()
embedding_model=MXBAIEmbedding()
rag_pipeline = RAGPipeline(embedding_model)

@app.get("/")
async def root():
    return {"message": "bookipedia"}

@app.get("/stream_response_and_sources")
async def stream_response_and_sources(user_prompt: str,
                                    chat_summary: str,
                                    chat: str,
                                    book_ids: Annotated[list[str] | None, Query()],
                                    enable_web_retrieval:bool = True):
    # Initialize RAG pipeline
    async def stream_generator():
        # Yield data stream
        async for chunk in rag_pipeline.generate_answer(user_prompt, chat_summary, chat, book_ids, enable_web_retrieval):
            yield chunk.encode('utf-8')
        # Yield metadata as first part of the stream
        yield b'\n\nSources: '
        yield json.dumps(rag_pipeline.metadata).encode('utf-8') + b'\n'
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/chat_summary")
async def chat_summary(response: str, user_prompt: str, prev_summary:str):
    summary = rag_pipeline.generate_chat_summary(response, user_prompt, prev_summary)
    summary_json = json.dumps({"summary": summary}).encode('utf-8')
    return summary_json

@app.get("/synthesize_audio_from_text/")
async def synthesize_audio_endpoint(text: str, speed: float = 1):
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        lines = text.split('\n')
        for line in lines:
            audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
            for audio_bytes in audio_stream:
                yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/x-wav")

@app.get("/synthesize_audio_from_pages/")
async def synthesize_audio_from_file_endpoint(book_id: str, pages: Annotated[list[int] | None, Query()], speed: float = 1):
    def synthesize_audio():
        # Split the text into lines and synthesize each line
        for page in pages:
            text = rag_pipeline.get_page_text(book_id, page)
            lines = text.split('\n')
            for line in lines:
                audio_stream = voice.synthesize_stream_raw(line, length_scale= 1/speed)
                for audio_bytes in audio_stream:
                    yield audio_bytes
    return StreamingResponse(synthesize_audio(), media_type="audio/x-wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
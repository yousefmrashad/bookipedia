from root_config import *
from utils.init import *
# ================================================== #
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from RAG.rag_pipeline import RAGPipeline
from preprocessing.embeddings_class import MXBAIEmbedding
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

@app.get("/stream_response_and_summary")
async def stream_response_and_summary(user_prompt: str,
                                    chat_summary: str,
                                    chat: str,
                                    book_ids: list[str] = None,
                                    enable_web_retrieval=True):
    # Initialize RAG pipeline
    async def stream_generator():
        # Yield data stream
        response = ''
        async for chunk in rag_pipeline.generate_answer(user_prompt, chat_summary, chat, book_ids, enable_web_retrieval):
            response += chunk 
            yield chunk.encode('utf-8')

        # Yield metadata as first part of the stream
        yield b'\n'
        yield json.dumps(rag_pipeline.metadata).encode('utf-8') + b'\n'
        yield json.dumps(rag_pipeline.generate_chat_summary(response, user_prompt, chat)).encode('utf-8') + b'\n'
    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.get("/synthesize_audio/")
async def synthesize_audio_endpoint(text: str):
    def synthesize_audio(text: str):
        # Split the text into lines and synthesize each line
        lines = text.split('\n')
        for line in lines:
            audio_stream = voice.synthesize_stream_raw(line)
            for audio_bytes in audio_stream:
                yield audio_bytes
                
    return StreamingResponse(synthesize_audio(text), media_type="audio/x-wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

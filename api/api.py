# Utils
from root_config import *
from utils.init import *

# API
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# TTS
from piper import PiperVoice

from utils.db_config import DB
from preprocessing.document import Document
from preprocessing.embedding import HFEmbedding
from rag.rag_pipeline import RAGPipeline
# ===================================================================== #

# Schemas
class ChatParams(BaseModel):
    user_prompt: str
    chat_summary: str
    chat: str
    doc_ids: list[str] = None

class TTSText(BaseModel):
    text: str
# --------------------------------------------------------------------- #

# Initializations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

voice = PiperVoice.load(PIPER_MODEL_PATH,
                        PIPER_CONFIG_PATH,
                        use_cuda=False)


app = FastAPI(
    title="Bookipedia AI Server",
    description="Bookipedia AI Server is an AI inference server for the Bookipedia application, which serves as an online library with an AI-powered reading assistant. The server utilizes state-of-the-art language models (LLMs), optical character recognition (OCR), and text-to-speech (TTS) features.",
    version="0.0.4"
)
embedding_model = HFEmbedding()
client = DB().connect()
rag_pipeline = RAGPipeline(embedding_model, client)
# --------------------------------------------------------------------- #

# Background Tasks
def process_document(doc: Document, doc_id: str):
    logger.info(f"Processing document {doc_id}")
    try:
        if (doc.text_based):
            doc.process_document(embedding_model, client)
            requests.patch(ACKNOWLEDGE_URL + doc.doc_id, json={"message": "Document preprocessing completed."})
            logger.info(f"Document {doc_id} preprocessed successfully.")
        else:
            doc.get_text_based_document()
            with open(doc.doc_path, 'rb') as file:
                response = requests.post(POST_HOCR_URL + doc_id, files={'file': (file.name, file, 'application/pdf')})
            if (response.status_code == 202):
                doc.doc_id = response.json()["file_id"]
                doc.process_document(embedding_model, client)
                requests.patch(ACKNOWLEDGE_URL + doc.doc_id, json={"message": "Document OCR and preprocessing completed."})
                logger.info(f"HOCR file for document {doc_id} posted and processed successfully.")
            else:
                logger.error(f"Failed to post HOCR file for document {doc_id}. Status code: {response.status_code}, Text: {response.text}")
        os.remove(doc.doc_path)
    except Exception as e:
        logger.exception(f"Exception occurred while processing document {doc_id}: {str(e)}")
# ---------------------------------------------- #

def summarize_chat(chat_id: str, response: str, user_prompt: str, prev_summary: str):
    logger.info(f"Summarizing chat {chat_id}")
    try:
        chat_summary = rag_pipeline.generate_chat_summary(response, user_prompt, prev_summary)
        response = requests.patch(CHAT_SUMMARY_URL + chat_id, json={"chat_summary": chat_summary})
        if response.status_code == 202:
            logger.info(f"Chat summary for chat {chat_id} updated successfully.")
        else:
            logger.error(f"Failed to update chat summary for chat {chat_id}. Status code: {response.status_code}")
    except Exception as e:
        logger.exception(f"Exception occurred while summarizing chat {chat_id}: {str(e)}")
# -------------------------------------------------- #

# Endpoints
@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "bookipedia"}
# -------------------------------------------------- #

@app.post("/add_document/{doc_id}")
async def add_document(background_tasks: BackgroundTasks, doc_id: str, url: str, lib_doc: bool = False):
    logger.info(f"Add document endpoint called with doc_id: {doc_id}, url: {url}, lib_doc: {lib_doc}")
    try:
        response = requests.get(url, stream=True)
        if (response.status_code == 200):
            doc_path = doc_id + '.pdf'
            with open(doc_path, 'wb') as file:
                file.write(response.content)
            logger.info(f"File {doc_path} downloaded successfully.")
        else:
            logger.error(f"Failed to download file from {url}. Status code: {response.status_code}")
            return {"message": "Failed to download file.", "status code": response.status_code}

        doc = Document(doc_path, doc_id, lib_doc)
        background_tasks.add_task(process_document, doc, doc_id)
        if (doc.text_based):
            logger.info(f"Document {doc_id} is text-based. Preprocessing started.")
            return {"message": "Document is text-based. Preprocessing started.", "OCR": False}
        else:
            logger.info(f"Document {doc_id} is not text-based. Applying OCR.")
            return {"message": "Document is not text-based. Applying OCR.", "OCR": True}
    except Exception as e:
        logger.exception(f"Exception occurred while adding document {doc_id}: {str(e)}")
        return {"message": "An error occurred while adding the document."}
# -------------------------------------------------- #

@app.delete("/delete_document/{doc_id}")
async def delete_document(doc_id: str):
    logger.info(f"Delete document endpoint called with doc_id: {doc_id}")
    try:
        rag_pipeline.db.delete(doc_id)
        logger.info(f"Document {doc_id} deleted successfully.")
        return {"message": "Document deleted successfully."}
    except Exception as e:
        logger.exception(f"Exception occurred while deleting document {doc_id}: {str(e)}")
        return {"message": "An error occurred while deleting the document."}
# -------------------------------------------------- #

@app.get("/chat_response/{chat_id}")
async def chat_response(background_tasks: BackgroundTasks,
                        chat_id: str,
                        chat_params: ChatParams,
                        enable_web_retrieval: bool = False):
    logger.info(f"Chat response endpoint called with chat_id: {chat_id}, enable_web_retrieval: {enable_web_retrieval}")
    try:
        # Extract parameters
        user_prompt = chat_params.user_prompt
        chat_summary = chat_params.chat_summary
        chat = chat_params.chat
        doc_ids = chat_params.doc_ids

        # Generate retrieval method & retrieval query
        retrieval_method, retrieval_query = rag_pipeline.generate_retrieval_query(user_prompt, chat_summary)

        # Generate context
        context, metadata = rag_pipeline.generate_context(retrieval_method, retrieval_query, doc_ids, enable_web_retrieval)

        # Response generator
        async def stream_generator():
            answer = rag_pipeline.generate_answer(user_prompt, chat, context)

            # Yield data stream
            response = ""
            async for chunk in answer:
                response += chunk
                yield chunk.encode("utf-8")

            yield b'\n\n{[sources]\n\n"sources": '
            yield json.dumps(metadata).encode("utf-8") + b"}"
            
            # Add chat summary to background tasks
            background_tasks.add_task(summarize_chat, chat_id, response, user_prompt, chat_summary)

        logger.info(f"Generated context and response for chat_id: {chat_id}")
        return StreamingResponse(stream_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.exception(f"Exception occurred while generating chat response for chat_id {chat_id}: {str(e)}")
        return {"message": "An error occurred while generating the chat response."}
# -------------------------------------------------- #

@app.get("/summarize_pages/{doc_id}")
async def summarize_pages(doc_id: str, pages: Annotated[list[int], Query()]):
    logger.info(f"Summarize pages endpoint called with doc_id: {doc_id}, pages: {pages}")
    try:
        async def stream_generator():
            summary = await rag_pipeline.summarize_pages(doc_id, pages)
            async for chunk in summary:
                yield chunk.encode("utf-8")
                
        logger.info(f"Summarization for document {doc_id} pages {pages} started")
        return StreamingResponse(stream_generator(), media_type="text/plain")
    
    except Exception as e:
        logger.exception(f"Exception occurred in summarizing pages for doc_id {doc_id}: {str(e)}")
        return {"message": "An error occurred during the summarization of the specified pages."}
# -------------------------------------------------- #

@app.get("/tts/")
async def text_to_speech(tts_text: TTSText, speed: float = 1):
    logger.info(f"TTS endpoint called with text: {tts_text.text}, speed: {speed}")
    try:
        text = tts_text.text
        def synthesize_audio():
            lines = text.split('\n')
            for line in lines:
                audio_stream = voice.synthesize_stream_raw(line, length_scale=1/speed)
                for audio_bytes in audio_stream:
                    yield audio_bytes

        logger.info("TTS synthesis started")
        return StreamingResponse(synthesize_audio(), media_type="audio/raw")
    except Exception as e:
        logger.exception(f"Exception occurred in TTS synthesis: {str(e)}")
        return {"message": "An error occurred during TTS synthesis."}
# -------------------------------------------------- #

@app.get("/tts_pages/{doc_id}")
async def pages_to_speech(doc_id: str, pages: Annotated[list[int], Query()], speed: float = 1):
    logger.info(f"TTS pages endpoint called with doc_id: {doc_id}, pages: {pages}, speed: {speed}")
    try:
        def synthesize_audio():
            for page in pages:
                text = rag_pipeline.get_page_text(doc_id, page)
                lines = text.split('\n')
                for line in lines:
                    audio_stream = voice.synthesize_stream_raw(line, length_scale=1/speed)
                    for audio_bytes in audio_stream:
                        yield audio_bytes

        logger.info(f"TTS synthesis for document {doc_id} pages {pages} started")
        return StreamingResponse(synthesize_audio(), media_type="audio/raw")
    except Exception as e:
        logger.exception(f"Exception occurred in TTS pages synthesis for doc_id {doc_id}: {str(e)}")
        return {"message": "An error occurred during TTS synthesis for the specified pages."}
# -------------------------------------------------- #

if (__name__ == "__main__"):
    import uvicorn
    logger.info("Starting Bookipedia AI Server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
# --------------------------------------------------------------------- #

from pydantic import BaseModel

# Request body definitions
class ChatParams(BaseModel):
    user_prompt: str
    chat_summary: str
    chat: str
    doc_ids: list[str] = None

class TTSText(BaseModel):
    text: str
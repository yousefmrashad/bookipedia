# Utils
from root_config import *
from utils.init import *
# ================================================== #

# HFEmbedding Model
class HFEmbedding(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME,
                prompt=RETRIEVAL_PROMPT):
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        self.prompt = prompt

    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str):
        text = self.prompt + text if (self.prompt) else text
        return self.model.encode(text).squeeze().tolist()
# -------------------------------------------------- #
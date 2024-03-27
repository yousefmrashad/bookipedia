# Utils
from root_config import *
from utils.init import *
# ================================================== #

# AnglEEmbedding Model
class AnglEEmbedding(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME,
                prompt=Prompts.C, pooling_strategy="cls"):
        
        self.model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy).cuda()
        self.prompt = prompt
        
        if (self.prompt):
            self.model.set_prompt(prompt=prompt)

    def embed_documents(self, texts: list[str]):
        texts = [{"text": text} for text in texts] if (self.prompt) else texts
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str):
        text = {"text": text} if (self.prompt) else text
        return self.model.encode(text).squeeze().tolist()
# -------------------------------------------------- #
    
# MXBAIEmbedding Model
class MXBAIEmbedding(Embeddings):
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1",
                prompt=RETRIEVAL_PROMPT):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name).cuda()
        self.prompt = prompt

    def embed_documents(self, texts: list[str]):
        texts = [text for text in texts]
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str):
        text = self.prompt + text if (self.prompt) else text
        return self.model.encode(text).squeeze().tolist()
# -------------------------------------------------- #
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

    def embed_documents(self, texts: list[str]):
        self.model.set_prompt(None)
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str):
        self.model.set_prompt(prompt=self.prompt)
        text = {"text": text}
        return self.model.encode(text).squeeze().tolist()
# -------------------------------------------------- #
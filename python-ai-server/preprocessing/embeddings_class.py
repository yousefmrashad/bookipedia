# Utils
from utils_config import *
from utils import *
# ================================================== #

# AnglEEmbedding Model
class AnglEEmbedding(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME,
                prompt=Prompts.C, pooling_strategy="cls"):
        
        self.model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy).cuda()
        self.prompt = prompt
        
        if (self.prompt):
            self.model.set_prompt(prompt=prompt)

    def embed_documents(self, texts):
        texts = [{"text": text} for text in texts] if (self.prompt) else texts
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        text = {"text": text} if (self.prompt) else text
        return self.model.encode(text).squeeze().tolist()
# -------------------------------------------------- #
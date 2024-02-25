from utils_config import *
from utils import *
# -------------------------------------------------------------------- #

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
        return self.model.encode(texts)

    def embed_query(self, text):
        text = {"text": text} if (self.prompt) else text
        return self.model.encode(text).squeeze()
# -------------------------------------------------------------------- #

class Document:
    def __init__(self, doc_path: str, doc_id: int):
        self.doc_path = doc_path
        self.doc_id = doc_id
    
    def load_and_split(self, chunk_size=CHUNCK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=count_tokens,
                                                    separators=separators,
                                                    is_separator_regex=True)
        
        self.chunks = PyPDFLoader(self.doc_path).load_and_split(text_splitter)
            
    def chunks_to_embeddings(self):
        pass
        

    def save_to_database(self):
        pass

    def preprocess(self):
        self.load_and_split()
# -------------------------------------------------------------------- #

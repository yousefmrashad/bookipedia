from utils_config import *
from utils import *
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
        
    def to_vectorstore(self, embedder: Embeddings, url: str = "http://localhost:8080"):
        self.vectorstore = Weaviate.from_documents(self.chunks, embedder, weaviate_url = url)

    def preprocess(self):
        self.load_and_split()
# -------------------------------------------------------------------- #
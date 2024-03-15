from utils_config import *
from utils import *
# -------------------------------------------------------------------- #

class Document:
    def __init__(self, doc_path: str, doc_id: str):
        self.doc_path = doc_path
        self.doc_id = doc_id
    
    def load_and_split(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=count_tokens,
                                                    separators=separators,
                                                    is_separator_regex=True)
        
        self.chunks = PyPDFLoader(self.doc_path).load_and_split(text_splitter)
        
        for c in self.chunks:
            c.metadata['source_id'] = self.doc_id

    def generate_embeddings(self, embedder: Embeddings):
        self.embeddings = embedder.embed_documents([chunk.page_content for chunk in self.chunks])
    
    def store_in_db(self, client: WeaviateClient = None):
        if client is None:
            client = weaviate.connect_to_local()
        
        cls = client.collections.get("am_chunks")
        fours = 0 
        sixteens = 0
        for i, c in enumerate(self.chunks):
            if i != 0 and i % 4 == 0:
                fours += 1
            if i != 0 and i % 16 == 0:
                sixteens += 1
            cls.data.insert(
                properties={
                    "source_id": c.metadata["source_id"],
                    "page_no": c.metadata["page"] + 1,
                    "text": c.page_content,
                    "child": fours,
                    "parent": sixteens
                },
                vector=self.embeddings[i]) 

    def preprocess(self):
        self.load_and_split()
# -------------------------------------------------------------------- #
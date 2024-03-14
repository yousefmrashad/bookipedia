# Utils
from utils_config import *
from utils import *
# ================================================== #

class Document:
    def __init__(self, doc_path: str, doc_id: str):
        self.doc_path = doc_path
        self.doc_id = doc_id
    # -------------------------------------------------- #
    
    def load_and_split(self, chunk_size=CHUNCK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                        length_function=count_tokens, separators=separators,
                                                        is_separator_regex=True)
        
        self.chunks = PyPDFLoader(self.doc_path).load_and_split(text_splitter)
        
        for chunk in self.chunks:
            chunk.metadata["source_id"] = self.doc_id
    # -------------------------------------------------- #

    def generate_embeddings(self, embedder: Embeddings):
        self.embeddings = embedder.embed_documents([chunk.page_content for chunk in self.chunks])
    # -------------------------------------------------- #
    
    def store_in_db(self, client: WeaviateClient = None):
        client = weaviate.connect_to_local() if (client is None) else client
        
        objs = []
        for i, chunk in enumerate(self.chunks):
            properties = {
                "source_id": chunk.metadata["source_id"],
                "page_no": chunk.metadata["page"],
                "text": chunk.page_content
            }
            obj = weaviate.classes.data.DataObject(properties, vector=self.embeddings[i])
            objs.append(obj)
            
        collection = client.collections.get("Chunks")
        collection.data.insert_many(objs)
    # -------------------------------------------------- #
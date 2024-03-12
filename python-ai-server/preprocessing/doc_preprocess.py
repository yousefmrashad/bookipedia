from utils_config import *
from utils import *
# -------------------------------------------------------------------- #

class Document:
    def __init__(self, doc_path: str, doc_id: str, client: WeaviateClient = None):
        self.doc_path = doc_path
        self.doc_id = doc_id
        if client:
            self.client = client
        else:
            self.client = weaviate.connect_to_local()
    
    def load_and_split(self, chunk_size=CHUNCK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS):

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
    
    def store_in_db(self):
        objs = list()
        for i, c in enumerate(self.chunks):
            objs.append(classes.data.DataObject(
                properties={
                    "source_id": c.metadata["source_id"],
                    "page_no": c.metadata["page"],
                    "text": c.page_content
                },
                vector=self.embeddings[i]
            ))

        collection = self.client.collections.get("Chunks")
        collection.data.insert_many(objs)    # This uses batching under the hood

    def preprocess(self):
        self.load_and_split()
# -------------------------------------------------------------------- #
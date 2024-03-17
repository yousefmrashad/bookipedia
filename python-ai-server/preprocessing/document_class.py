# Utils
from root_config import *
from utils.init import *
# ================================================== #

class Document:
    def __init__(self, doc_path: str, doc_id: str):
        self.doc_path = doc_path
        self.doc_id = doc_id
    # -------------------------------------------------- #
    
    def load_and_split(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS):

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
    
    def store_in_db(self, client: WeaviateClient):        
        objs = []
        l1, l2 = (0, 0)
        # Check if source id already exists
        collection = client.collections.get(DB_NAME)
        exsit_filter = wvc.query.Filter.by_property("source_id").equal(self.doc_id)
        
        if (len((collection.query.fetch_objects(filters=exsit_filter, limit=1).objects)) == 0):
            for i, chunk in enumerate(self.chunks):
                if (i != 0) and (i % L1 == 0): l1 += 1
                if (i != 0) and (i % L2 == 0): l2 += 1
                collection.data.insert(
                properties = {
                    "source_id": chunk.metadata["source_id"],
                    "page_no": chunk.metadata["page"]+1,
                    "text": chunk.page_content,
                    "l1": l1,
                    "l2": l2
                },
                vector=self.embeddings[i])
    # -------------------------------------------------- #
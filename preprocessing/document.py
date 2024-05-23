# Utils
from root_config import *
from utils.init import *
# ================================================== #

class Document:
    def __init__(self, doc_path: str, doc_id: str, lib_doc=False):
        self.doc_path = doc_path
        self.doc_id = doc_id
        self.doc = fitz.open(doc_path)

        if (lib_doc):
            self.text_based = True
        else:
            self.text_based = calculate_imagebox_percentage(self.doc) < 0.5
    # -------------------------------------------------- #
    
    def get_text_based_document(self):
        from preprocessing.ocr import OCR
        OCR(self.doc_path).apply_ocr()
    # -------------------------------------------------- #
    
    def load_and_split(self):
        if(self.text_based):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                            length_function=count_tokens, separators=MD_SEPARATORS,
                                                            is_separator_regex=True)
            md_texts = []
            metadatas = []
            for i in range(self.doc.page_count):
                metadata = {}
                md_texts.append(to_markdown(self.doc, [i]))
                metadata['source_id'] = self.doc_id
                metadata['page'] = i
                metadatas.append(metadata)
            self.chunks = text_splitter.create_documents(md_texts, metadatas)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                            length_function=count_tokens, separators=SEPARATORS,
                                                            is_separator_regex=True)
            self.chunks = PyPDFLoader(self.doc_path).load_and_split(text_splitter)
        
        for chunk in self.chunks:
            chunk.metadata["source_id"] = self.doc_id
            
        # Filtering chunks
        self.chunks = [c for c in self.chunks if (re.search(r"[a-zA-Z]", c.page_content))]
    # -------------------------------------------------- #

    def generate_embeddings(self, embedder: Embeddings):
        self.embeddings = embedder.embed_documents([chunk.page_content for chunk in self.chunks])
    # -------------------------------------------------- #
    
    def store_in_db(self, client: WeaviateClient):        
        collection = client.collections.get(DB_NAME)

        # Check if the source id already exists
        exist_filter = wvc.query.Filter.by_property("source_id").equal(self.doc_id)
        not_exist = (len((collection.query.fetch_objects(filters=exist_filter, limit=1).objects)) == 0)

        if (not_exist):
            objs = []
            l1, l2 = (0, 0)
            for i, chunk in enumerate(self.chunks):
                if (i != 0) and (i % L1 == 0): l1 += 1
                if (i != 0) and (i % L2 == 0): l2 += 1
            
                properties = {
                    "index": i,
                    "source_id": chunk.metadata["source_id"],
                    "page_no": chunk.metadata["page"]+1,
                    "text": chunk.page_content,
                    "l1": l1,
                    "l2": l2
                }
                obj = wvc.data.DataObject(properties=properties, vector=self.embeddings[i])
                objs.append(obj)

            collection.data.insert_many(objs)
    # -------------------------------------------------- #
    
    def process_document(self, embedder: Embeddings, client: WeaviateClient):       
        self.load_and_split()
        self.generate_embeddings(embedder)
        self.store_in_db(client)
    # -------------------------------------------------- #
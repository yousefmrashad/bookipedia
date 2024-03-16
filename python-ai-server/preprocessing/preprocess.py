# Utils
from root_config import *
from utils.init import *

# Modules
from preprocessing.document_class import Document
from preprocessing.embeddings_class import AnglEEmbedding
# ================================================== #

class DocumentPreprocess:
    def __init__(self, doc_path: str, doc_id: str):
        self.doc_path = doc_path
        self.doc_id = doc_id

    def get_text_based_document(self):
        pages = pypdf.PdfReader(self.doc_path).pages
        for page in pages:
            if (page.extract_text().strip()):
                return
    
        from preprocessing.ocr import OCR
        hocr_doc_path = OCR(self.doc_path).apply_ocr()
        self.doc_path = hocr_doc_path

    def preprocess(self, client: WeaviateClient):
        document = Document(self.doc_path, self.doc_id)
        document.load_and_split()
        document.generate_embeddings(embedder=AnglEEmbedding())
        document.store_in_db(client)
# -------------------------------------------------- #
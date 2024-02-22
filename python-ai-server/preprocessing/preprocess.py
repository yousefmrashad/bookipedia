from utils_config import *
from utils import *

from doc_preprocess import DocLoader
# -------------------------------------------------------------------- #

DOC_PATH = ""
DOC_ID = 1
# -------------------------------------------------------------------- #

def get_text_document(doc_path: str):
    pages = PyPDF2.PdfReader(doc_path).pages
    for page in pages:
        if (page.extract_text().strip()):
            return
    
    from ocr import OCR
    OCR(doc_path).apply_ocr()
# -------------------------------------------------------------------- #
    
loader = DocLoader(DOC_PATH)
docs = loader.load()
print(docs)
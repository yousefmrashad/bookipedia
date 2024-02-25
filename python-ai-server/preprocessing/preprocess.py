from utils_config import *
from utils import *

from doc_preprocess import Document
# -------------------------------------------------------------------- #

# DOC_PATH = ""
DOC_PATH = r""
DOC_ID = 1
# -------------------------------------------------------------------- #

def text_based_document(doc_path: str):
    pages = pypdf.PdfReader(doc_path).pages
    for page in pages:
        if (page.extract_text().strip()):
            return
    
    from ocr import OCR
    OCR(doc_path).apply_ocr()
# -------------------------------------------------------------------- #

# Make sure that the Document is text-based document
text_based_document(DOC_PATH)

# Prepare the document
document = Document(DOC_PATH, DOC_ID)
document.preprocess()
from utils_config import *
from utils import *

from doc_preprocess import Document
# -------------------------------------------------------------------- #

# DOC_PATH = ""
doc_path = r""
doc_id = 1
# ------------------------------------------------------------------- #

def text_based_document(doc_path: str):
    pages = pypdf.PdfReader(doc_path).pages
    for page in pages:
        if (page.extract_text().strip()):
            return
    
    from ocr import OCR
    OCR(doc_path).apply_ocr()
    doc_path = doc_path.replace(".pdf", "_hocr.pdf")
# -------------------------------------------------------------------- #

# Make sure that the Document is text-based document
text_based_document(doc_path)

# Prepare the document
document = Document(doc_path, doc_id)
document.preprocess()
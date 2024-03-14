# Utils
from utils_config import *
from utils import *

# Modules
from doc_preprocess import Document
from embeddings_class import AnglEEmbedding
# ================================================== #

doc_path = r""
doc_id = "بيقولك مرة واحد عنده زهايمر اتاوب ف نسي بوقه مفتوح"
# -------------------------------------------------- #

def get_text_based_document(doc_path: str):
    pages = pypdf.PdfReader(doc_path).pages
    for page in pages:
        if (page.extract_text().strip()):
            return doc_path
    
    from ocr import OCR
    hocr_doc_path = OCR(doc_path).apply_ocr()

    return hocr_doc_path
# -------------------------------------------------- #

# Make sure that the Document is text-based document
doc_path = get_text_based_document(doc_path)

# Prepare the document
document = Document(doc_path, doc_id)
document.load_and_split()
# document.generate_embeddings(embedder=AnglEEmbedding())
# document.store_in_db()
# -------------------------------------------------- #
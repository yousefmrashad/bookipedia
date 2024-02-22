from utils_config import *
from utils import *
# -------------------------------------------------------------------- #

class DocLoader:
    def __init__(self, doc_path: str, doc_id: int):
        self.doc_path = doc_path
    
    def load(self, chunk_size=256, chunk_overlap=24, separators=["\n\n", "(?<=\w{2}\.\s)", "\n"]):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=count_tokens,
                                                    separators=separators,
                                                    is_separator_regex= True)
        
        self.docs = PyPDFium2Loader(self.doc_path).load_and_split(text_splitter)
            
    def docs_to_embeddings(self):
        pass

    def save_to_database(self):
        pass
# -------------------------------------------------------------------- #

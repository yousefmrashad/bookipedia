# Utils
from root_config import *
from utils.init import *
from utils.db_config import DB

# Modules
from preprocessing.preprocess import DocumentPreprocess
from preprocessing.embeddings_class import AnglEEmbedding
from RAG.weaviate_class import Weaviate
# ================================================== #

doc_path = r"C:\Users\LEGION\Desktop\Transformer_Network.pdf"
doc_id = "LionelMessi181269"
# doc_id = "LionelMessi1812"

client = DB().connect()
db = Weaviate(client, embedder=AnglEEmbedding())

# print("Processing Document...")
# DocumentPreprocess(doc_path, doc_id).preprocess(client)

print("Performing some shit...")
q = "What is positional encoding in transformers?"
docs = db.similarity_search(query=q, source_ids=[doc_id], k=20)

print("="*50)
for doc, score in docs:
    print(f"Page {doc.metadata['page_no']} | Score {score}")
    print(doc.page_content)
    print("="*50)

client.close()
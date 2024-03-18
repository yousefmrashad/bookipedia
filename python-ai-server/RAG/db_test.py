# Utils
from root_config import *
from utils.init import *
from utils.db_config import DB

# Modules
from preprocessing.document_class import Document
from preprocessing.embeddings_class import AnglEEmbedding
from RAG.weaviate_class import Weaviate
# ================================================== #

doc_path = r"C:\Users\LEGION\Desktop\Transformer_Network.pdf"
doc_id = "transformer-book69"

client = DB().connect()
db = Weaviate(client, embedder=AnglEEmbedding())

print("Deleting...") 
db.delete(doc_id)

print("Processing...")
document = Document(doc_path, doc_id)
document.preprocess(client)

q = "Give me the introduction of this book"
docs = db.similarity_search(query=q, source_ids=[doc_id])

print("="*50)
for doc, score in docs:
    print(f"Page {doc.metadata['page_no']} | Score {score}")
    print(doc.page_content)
    print("="*50)

client.close()

# f_obj = client.collections.get(DB_NAME).query.fetch_objects(limit=1, sort=SORT).objects[0]
# print(f_obj.properties["page_no"])
# print(f_obj.properties["text"])
# client.close()
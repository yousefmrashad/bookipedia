# Utils
from root_config import *
from utils.init import *

from utils.db_config import DB
from preprocessing.document import Document
from preprocessing.embedding import HFEmbedding
from rag.rag_pipeline import RAGPipeline
# ===================================================================== #

# Show results
async def print_result(results: AsyncIterator[str]):
    async for result in results:
        print(result, end="", flush=True)
# --------------------------------------------------------------------- #

# Initail shit
embedding_model = HFEmbedding()
client = DB().connect()
rag_pipeline = RAGPipeline(embedding_model, client)
# --------------------------------------------------------------------- #

# Load a fucking document
doc_path = r"C:\Users\LEGION\Desktop\Transformer_Network.pdf"
doc_id = "69"
# doc = Document(doc_path, doc_id, lib_doc=True)
# doc.process_document(embedder=embedding_model, client=client)
# --------------------------------------------------------------------- #

# # Test that CRAPPY pipeline
# prompt = "what is the definition of a transforemer?"
# chat_summary = ""

# retrieval_method, retrieval_query = rag_pipeline.generate_retrieval_query(user_prompt=prompt, chat_summary=chat_summary)
# retrieval_method = "Vectorstore"

# print(f"Method: {retrieval_method}")
# print(f"Query: {retrieval_query}")

# context, metadata = rag_pipeline.generate_context(retrieval_method, retrieval_query, doc_ids=[doc_id])
# print("="*69)
# print(context)
# print("="*69)

# async def main():
#     answer = rag_pipeline.generate_answer(user_prompt=prompt, chat_summary=chat_summary, context=context)
#     await print_result(answer)
# asyncio.run(main())
# --------------------------------------------------------------------- #

# # Test summary pipeline
# async def summary():
#     s = await rag_pipeline.summarize_pages(doc_id, [1, 2])
#     await print_result(s)
# asyncio.run(summary())

client.close()
# --------------------------------------------------------------------- #
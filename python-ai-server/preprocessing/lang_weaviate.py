import weaviate.classes as wvc
from weaviate import WeaviateClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from weaviate.collections.classes.internal import QueryReturn, ReturnProperties
from langchain.vectorstores import VectorStore
import numpy as np
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def rerank_docs(
        query: str ,
        docs: list[Document],
        top_k :int) -> list[Document]:
    """Re-ranks a list of documents based on their relevance to a given query using a pre-trained sequence classification model.

    Args:
        - query (str): The query string used to determine the relevance of the documents.
        - docs (list[Document]): A list of Document objects to be re-ranked. Each Document object should have a `page_content` attribute containing the text of the document.
        - top_k (int): The number of top-ranked documents to return.


    Returns:
        - list[Document]: A list of the top-ranked Document objects, sorted by relevance to the query.
    """
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
    model.eval()
    # Prepare the query-document pairs for the model
    pairs = [[query , doc.page_content] for doc in docs]
    # Tokenize the pairs and generate scores
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # Sort the documents based on the scores and return the top-k documents
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_docs = [docs[i] for i in sorted_indices]
    return sorted_docs[: top_k]


def to_docs(response: QueryReturn[ReturnProperties, None]) -> list[Document]:
    docs = []
    for o in response.objects:
        text = o.properties['text']
        mets = {key: value for key, value in o.properties.items() if key != 'text'}
        if o.metadata.distance is not None:
            distance = o.metadata.distance
            docs.append((Document(page_content=text, metadata=mets), distance))
        else:
            docs.append(Document(page_content=text, metadata=mets))
    return docs

def sim_search(
        client:WeaviateClient,
        emb: Embeddings,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        source_ids: list = None) -> list[Document]:
    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None
    response = collection.query.hybrid(
    query= query,
    vector= query_emb,
    filters=filters,
    limit=k,
    alpha=alpha
    )
    docs = to_docs(response)
    return docs

def sim_search_score(
        client:WeaviateClient,
        emb: Embeddings,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        source_ids: list = None) -> list[tuple[Document, float]]:
    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None

    response = collection.query.hybrid(
    query= query,
    vector= query_emb,
    filters=filters,
    limit=k,
    alpha=alpha,
    return_metadata=wvc.query.MetadataQuery(distance=True)
    )

    docs = to_docs(response)
    return docs

def mmr_search(
        client:WeaviateClient,
        emb: Embeddings,
        query: str,
        source_ids: list = None,
        k: int = 5,
        alpha: float = 0.5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5) -> list[Document]:
    

    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None

    response = collection.query.hybrid(
    query= query,
    vector= query_emb,
    filters=filters,
    limit=fetch_k,
    alpha=alpha,
    return_metadata=wvc.query.MetadataQuery(distance=True),
    include_vector= True
    )
    embeddings = [o.vector['default'] for o in response.objects]
    mmr_selected = maximal_marginal_relevance(
            np.array(query_emb), embeddings, k=k, lambda_mult=lambda_mult
    )

    docs = []
    for idx in mmr_selected:
        text = response.objects[idx].properties['text']
        meta = {key: value for key, value in response.objects[idx].properties.items() if key != 'text'}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def delete(client: WeaviateClient, ids: list[str] = None) -> None:
    collection = client.collections.get("Chunks")
    if ids:
        collection.data.delete_many(
        where=wvc.query.Filter.by_id().contains_any(ids)  # Delete the 3 objects
    )
    else:
        raise ValueError("No ids provided to delete.")
    return None
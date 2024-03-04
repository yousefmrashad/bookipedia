import weaviate.classes as wvc
from weaviate import WeaviateClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from weaviate.collections.classes.internal import QueryReturn
import numpy as np

def to_docs(response: QueryReturn) -> list[Document]:
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
        source_ids: list = None) -> list[Document]:
    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None
    response = collection.query.near_vector(
    near_vector=query_emb,
    filters=filters,
    limit=k,
    return_properties=["text", "source_id", "page_no"]
    )
    docs = to_docs(response)
    return docs

def sim_search_score(
        client:WeaviateClient,
        emb: Embeddings,
        query: str,
        k: int = 5,
        source_ids: list = None) -> list[Document]:
    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None

    response = collection.query.near_vector(
    near_vector=query_emb,
    filters=filters,
    limit=k,
    return_metadata=wvc.query.MetadataQuery(distance=True),
    return_properties=["text", "source_id", "page_no"]
    )

    docs = to_docs(response)
    return docs

def mmr_search(
        client:WeaviateClient,
        emb: Embeddings,
        query: str,
        source_ids: list = None,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5) -> list[Document]:
    

    collection = client.collections.get("Chunks") 
    query_emb = emb.embed_query(query)

    filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if source_ids else None

    response = collection.query.near_vector(
    near_vector=query_emb,
    filters=filters,
    limit=fetch_k,
    return_metadata=wvc.query.MetadataQuery(distance=True),
    return_properties=["text", "source_id", "page_no"],
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
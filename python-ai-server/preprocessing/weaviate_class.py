# Utils
from utils_config import *
from utils import *
from typing import Any, List, Iterable
# ================================================== #

class Weaviate(VectorStore):
    def __init__(self, client: WeaviateClient, embedder: Embeddings) -> None:
        self.client = client
        self.embedder = embedder
    # -------------------------------------------------- #
    
    def add_texts(self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any) -> List[str]:
        return 
    # -------------------------------------------------- #

    def from_texts(self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any) -> List[Document]:
        return
    # -------------------------------------------------- #

    def to_docs(self, response: QueryReturn[ReturnProperties, None]) -> list[Document]:
        docs = []

        for obj in response.objects:
            text = obj.properties["text"]
            metadata = {key: value for key, value in obj.properties.items() if (key != "text")}
            docs.append((Document(page_content=text, metadata=metadata), obj.metadata.distance))
        
        return docs
    # -------------------------------------------------- #

    def similarity_search(self, query: str, k=5, alpha=0.5, source_ids=[]) -> list[Document]:
        
        collection = self.client.collections.get("Chunks") 
        query_emb = self.embedder.embed_query(query)

        filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if (source_ids) else None
        response = collection.query.hybrid(query=query, vector=query_emb, filters=filters, limit=k, alpha=alpha)
        docs = self.to_docs(response)

        return docs
    # -------------------------------------------------- #

    def similarity_search_with_score(self, query: str, k=5, alpha=0.5, source_ids=[]) -> list[tuple[Document, float]]:
        collection = self.client.collections.get("Chunks") 
        query_emb = self.embedder.embed_query(query)

        filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if (source_ids) else None
        response = collection.query.hybrid(query=query, vector=query_emb, filters=filters,
                                            limit=k, alpha=alpha, return_metadata=wvc.query.MetadataQuery(distance=True))
        docs = self.to_docs(response)

        return docs
    # -------------------------------------------------- #

    def max_marginal_relevance_search(self, query: str, k=5, alpha=0.5, fetch_k=20, lambda_mult=0.5, source_ids=[]) -> list[Document]:
        collection = self.client.collections.get("Chunks") 
        query_emb = self.embedder.embed_query(query)

        filters = wvc.query.Filter.by_property("source_id").contains_any(source_ids) if (source_ids) else None
        response = collection.query.hybrid(query=query, vector=query_emb, filters=filters, limit=fetch_k,
                                            alpha=alpha, return_metadata=wvc.query.MetadataQuery(distance=True),
                                            include_vector= True)
        
        embeddings = [obj.vector["default"] for obj in response.objects]
        mmr_selected = maximal_marginal_relevance(np.array(query_emb), embeddings, k=k, lambda_mult=lambda_mult)

        docs = []
        for idx in mmr_selected:
            text = response.objects[idx].properties["text"]
            metadata = {key: value for key, value in response.objects[idx].properties.items() if (key != "text")}
            docs.append(Document(page_content=text, metadata=metadata))
        
        return docs
    # -------------------------------------------------- #

    def delete(self, ids: list[str]):
        collection = self.client.collections.get("Chunks")
        collection.data.delete_many(where=wvc.query.Filter.by_id().contains_any(ids))
    # -------------------------------------------------- #

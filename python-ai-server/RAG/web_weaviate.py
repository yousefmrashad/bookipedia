# Utils
from root_config import *
from utils.init import *
# ================================================== #

class WebWeaviate(VectorStore):
    def __init__(self, client: WeaviateClient, embedder: Embeddings) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = self.client.collections.get('web_research')
    # -------------------------------------------------- #
        
    # -- Main Methods -- #
        
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] = None,
        **kwargs,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedder.embed_documents(texts)
        ids = []

        with self.collection.batch.dynamic() as batch:
            for i, metadatas in enumerate(metadatas):
                metadatas['text'] = texts[i]
                id = batch.add_object(
                    properties=metadatas,
                    vector=embeddings[i]
                )
                ids.append(id)
        return ids
    # -------------------------------------------------- #

    def similarity_search(self, query: str, k=5, alpha=0.9) -> list[Document]:
        query_emb = self.embedder.embed_query(query)

        objects = self.collection.query.hybrid(query=query, vector=query_emb,
                                                limit=k, alpha=alpha).objects
        docs = self.objects_to_docs(objects)
        return docs
    # -------------------------------------------------- #
    
    # -- Utility Methods -- #

    # Delete all documents in the collection
    def delete(self):
        response = self.collection.query.fetch_objects(limit=FETCHING_LIMIT, return_properties=[])
        ids = [o.uuid for o in response.objects]

        self.collection.data.delete_many(
            where=wvc.query.Filter.by_id().contains_any(ids)
        )
    # -------------------------------------------------- #
        
    # Response to documents
    def objects_to_docs(self, objects: list[Object]) -> list[Document]:
        docs = []
        for obj in objects:
            text = obj.properties["text"]
            metadata = {key: value for key, value in obj.properties.items() if key != "text"}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
    
    # -- Abstract Methods -- #
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs): return
    # -------------------------------------------------- #

import os

import weaviate
import weaviate.classes as wvc

from .config import DB_NAME

# ================================================== #


class DB:
    def __init__(self):
        self.client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", 8080)),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", 50051)),
        )

    def connect(self):
        if not (self.client.collections.exists(DB_NAME)):
            self.create()

        return self.client

    def create(self):
        self.client.collections.create(
            name=DB_NAME,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),
            properties=[
                wvc.config.Property(
                    name="index",
                    data_type=wvc.config.DataType.INT,
                    vectorize_property_name=False,
                ),
                wvc.config.Property(
                    name="source_id",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                ),
                wvc.config.Property(
                    name="page_no",
                    data_type=wvc.config.DataType.INT,
                    vectorize_property_name=False,
                ),
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                    tokenization=wvc.config.Tokenization.LOWERCASE,
                ),
                wvc.config.Property(
                    name="l1",
                    data_type=wvc.config.DataType.INT,
                    vectorize_property_name=False,
                ),
                wvc.config.Property(
                    name="l2",
                    data_type=wvc.config.DataType.INT,
                    vectorize_property_name=False,
                ),
            ],
        )


# -------------------------------------------------- #

# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# -------------------------------------------------- #

# -- Modules -- #

# Basics
import torch
from collections import Counter
import math

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
import PIL.Image
from pikepdf import Pdf
import re

# Document Loading
import fitz
from pdf4llm import to_markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from angle_emb import AnglE, Prompts
from sentence_transformers import SentenceTransformer

# Weaviate Class
import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.grpc import Sort
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.vectorstores.utils import maximal_marginal_relevance
from sentence_transformers import CrossEncoder
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# OpenAI
import tiktoken

# Type Hinting
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from weaviate import WeaviateClient
from weaviate.collections.classes.internal import QueryReturn, Object, ReturnProperties
# -------------------------------------------------- #

# -- Constants -- #

OPEN_AI_KEY = "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"

# OCR
DETECTION_MODEL = "db_mobilenet_v3_large"
RECOGNITION_MODEL = "crnn_mobilenet_v3_large"

# Document Load
CHUNK_SIZE = 128
CHUNK_OVERLAP = 32
MD_SEPARATORS = [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                r"(?<=\w{2}\.\s)",
                " ",
                "",
            ]
SEPARATORS = [r"(?<=\w{2}\.\s)", "\n"]

# Auto Merging
L1 = 4
L2 = 16
FETCHING_LIMIT = 1024

# Retrieving Filters
SORT = Sort.by_property(name="index", ascending=True)

# Embedding Model
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
RETRIEVAL_PROMPT = "Represent this sentence for searching relevant passages: " 

# Re-ranker Model
RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v1"

# Database Name
DB_NAME = "bookipedia"

# Back-End URLs
CHAT_SUMMARY_URL = "http://backend:3000/chat_summary"
POST_HOCR_URL = "http://backend:3000/post_hocr"
ACKNOWLEDGE_URL = "http://backend:3000/acknowledge"
# -------------------------------------------------- #
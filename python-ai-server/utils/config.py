# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# Root Path
import sys
root_path = os.path.dirname(os.path.dirname(__file__))
# -------------------------------------------------- #

# -- Modules -- #

# Basics
import torch
from collections import Counter
import math
import re 

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
import PIL.Image
from pikepdf import Pdf
import re, fitz

# Langchain
from langchain_community.document_loaders.pdf import PyMuPDFLoader
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
CHUNK_OVERLAP = 0
SEPARATORS = [r"(?<=\w{2}\.\s)", "\n"]

# Auto Merging
L1 = 4
L2 = 16
FETCHING_LIMIT = 1024

# Retrieving Filters
SORT = Sort.by_property(name="index", ascending=True)

# Embedding Model
# EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
RETRIEVAL_PROMPT = "Represent this sentence for searching relevant passages: " 

# Re-ranker Model
# RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v1"
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v1-turbo-en"

# Database Name
DB_NAME = "bookipedia"

# TTS Model Paths
PIPER_MODEL_PATH = os.path.join(root_path, "models/en_US-amy-medium.onnx")
PIPER_CONFIG_PATH = PIPER_MODEL_PATH + ".json"

# Back-End URLs
BACKEND_URL = "https://bookipedia-backend-pr-72.onrender.com/ai-api/"
# FILE_URL = BACKEND_URL + "file/"
CHAT_SUMMARY_URL = BACKEND_URL + "chat-summary/"
POST_HOCR_URL = BACKEND_URL + "ocr-file/"
ACKNOWLEDGE_URL = BACKEND_URL + "ai-applied/"
# -------------------------------------------------- #
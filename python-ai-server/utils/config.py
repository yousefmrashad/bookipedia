# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# -------------------------------------------------- #

# -- Modules -- #

# Basics
import torch
from collections import Counter

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
import PIL.Image
from pikepdf import Pdf
import re, pypdf

# Langchain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from angle_emb import AnglE, Prompts

# Weaviate Class
import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.grpc import Sort
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.vectorstores.utils import maximal_marginal_relevance
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# OpenAI
import tiktoken

# Type Hinting
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from weaviate import WeaviateClient
from weaviate.collections.classes.internal import QueryReturn, Object, ReturnProperties
# -------------------------------------------------- #

# -- Constants -- #

# OCR
DETECTION_MODEL = "db_mobilenet_v3_large"
RECOGNITION_MODEL = "crnn_mobilenet_v3_large"

# Document Load
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
SEPARATORS = [r"(?<=\w{2}\.\s\n)", r"(?<=\w{2}\.\s)", "\n"]

# Auto Merging
L1 = 4
L2 = 16
FETCHING_LIMIT = 1024

# Retrieving Filters
SORT = Sort.by_property(name="index", ascending=True)

# Embedding Model
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"

# Re-ranker Model
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Database Name
DB_NAME = "bookipedia"
# -------------------------------------------------- #
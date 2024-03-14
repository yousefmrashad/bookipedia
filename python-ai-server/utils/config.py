# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# -------------------------------------------------- #

# -- Modules -- #

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
import PIL.Image
from pikepdf import Pdf
import re, pypdf

# Langchain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from angle_emb import AnglE, Prompts

# Weaviate Class
import weaviate
import weaviate.classes as wvc
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

# OpenAI
import tiktoken

# Type Hinting
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from weaviate import WeaviateClient
from weaviate.collections.classes.internal import QueryReturn, ReturnProperties
# -------------------------------------------------- #

# -- Constants -- #

# OCR
DETECTION_MODEL = "db_mobilenet_v3_large"
RECOGNITION_MODEL = "crnn_mobilenet_v3_large"

# Document Load
CHUNCK_SIZE = 256
CHUNK_OVERLAP = 32
SEPARATORS = [r"(?<=\w{2}\.\s)", "\n"]

# Embedding Model
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
# -------------------------------------------------- #
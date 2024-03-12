# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# -------------------------------------------------------------------- #

# -- Modules -- #

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
from .hocr import HocrTransform
from pikepdf import Pdf
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element
import re, pypdf
from PIL.Image import fromarray

# Langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.embeddings import Embeddings
from angle_emb import AnglE, Prompts
from langchain_community.vectorstores.weaviate import Weaviate
import weaviate
from weaviate import WeaviateClient, classes

# OpenAI
import tiktoken
# -------------------------------------------------------------------- #

# -- Constants -- #

# OCR
DETECTION_MODEL = "db_mobilenet_v3_large"
RECOGNITION_MODEL = "crnn_mobilenet_v3_large"

# Document Load
CHUNK_SIZE = 128
CHUNK_OVERLAP = 32
SEPARATORS = [r"(?<=\w{2}\.\s)", "\n"]

# Embedding Model
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
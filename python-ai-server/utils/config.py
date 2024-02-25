# Using PyTorch
import os
os.environ["USE_TORCH"] = "1"
# -------------------------------------------------------------------- #

# -- Modules -- #

# Image Preprocessing
import numpy as np
import cv2 as cv

# HOCR
from reportlab.lib.colors import black
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

from math import atan, cos, sin
from typing import Dict, Optional, Tuple
import re, PIL.Image, pypdf, tempfile

# Langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.embeddings import Embeddings
from angle_emb import AnglE, Prompts

# OpenAI
import tiktoken
# -------------------------------------------------------------------- #

# -- Constants -- #

# OCR
DETECTION_MODEL = "db_mobilenet_v3_large"
RECOGNITION_MODEL = "crnn_mobilenet_v3_large"

# Document Load
CHUNCK_SIZE = 256
CHUNK_OVERLAP = 32
SEPARATORS = ["(?<=\w{2}\.\s)", "\n"]

# Embedding Model
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
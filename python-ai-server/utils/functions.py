# Config
from root_config import *
from utils.config import *

# ================================================== #

# -- Helpers -- #

# OCR
def map_values(img, in_min, in_max, out_min, out_max):
    return (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
# -------------------------------------------------- #

# Document Loader
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
# -------------------------------------------------- #

# Retrieving Filters
def id_filter(source_id: str):
    return wvc.query.Filter.by_property("source_id").equal(source_id)

def ids_filter(source_ids: list[str]):
    return wvc.query.Filter.by_property("source_id").contains_any(source_ids)

def page_filter(page_no: int):
    return wvc.query.Filter.by_property("page_no").equal(page_no)

# -------------------------------------------------- #

def merge_chunks(chunks: list[str]) -> str:
    if not chunks:
        return ""  # Return empty string if there are no chunks
    
    merged_text = chunks[0]  # Initialize merged text with the first chunk
    for chunk in chunks[1:]:
        if not chunk:  # Skip empty chunks
            continue
        
        overlap = find_overlap(merged_text, chunk)
        if overlap > 0:  # Ensure overlap is valid
            merged_text += chunk[overlap:]
        else:
            merged_text += " " + chunk  # If no overlap, append the entire chunk
    
    return merged_text


def find_overlap(text1: str, text2: str) -> int:
    if not text1 or not text2:
        return 0  # Return 0 if any input string is empty
    
    n = len(text1)
    m = len(text2)
    overlap = [0] * m
    j = 0

    for i in range(1, m):
        while j > 0 and text2[i] != text2[j]:
            j = overlap[j - 1]
        if text2[i] == text2[j]:
            j += 1
        overlap[i] = j

    j = 0
    for i in range(n):
        while j > 0 and text1[i] != text2[j]:
            j = overlap[j - 1]
        if text1[i] == text2[j]:
            j += 1
        if j == m:
            return m
    return j

def calculate_imagebox_percentage(doc: fitz.Document) -> float:
    total_imagebox_area = 0
    total_page_area = 0
    for page in doc:
        imagebox_area = 0
        page_area = page.rect.height * page.rect.width
        total_page_area += page_area
        for image in page.get_image_info():
            imagebox = image['bbox']
            imagebox_area += (imagebox[2] - imagebox[0]) * (imagebox[3] - imagebox[1])
            total_imagebox_area += imagebox_area
    return (total_imagebox_area / total_page_area)

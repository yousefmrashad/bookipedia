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
import cv2 as cv
import numpy as np
from tiktoken import get_encoding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.hocr import HocrTransform
import fitz

def map_values(img, in_min, in_max, out_min, out_max):
    '''
    Function to map values in range [in_min, in_max] to the range [out_min, out_max]
    '''
    return (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min

def filter_image(img: np.ndarray | None, kSize=55, whitePoint=120, blackPoint=70):
    # Applying high pass filter
    print("Applying high pass filter")
    
    if kSize % 2 == 0:
        kSize += 1
        
    kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
    filtered = cv.filter2D(img, -1, kernel)
    
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    
    print("Selecting white point...")
    _, img = cv.threshold(filtered, whitePoint, 255, cv.THRESH_TRUNC)
    
    img = map_values(img.astype('int32'), 0, whitePoint, 0, 255).astype('uint8')
    
    print("Adjusting black point for final output...")
    img = map_values(img.astype('int32'), blackPoint, 255, 0, 255)
    
    _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
    img = img.astype('uint8')

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    (l, a, b) = cv.split(lab)
    img = cv.add(cv.subtract(l, b), cv.subtract(l, a))
    sharpened = cv.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    
    print("\nDone.")
    
    return sharpened

def correct_skew(image):
    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Find the minimum area rectangle that encloses the contour
    rect = cv.minAreaRect(largest_contour)
    angle = rect[-1]

    # Rotate the image to correct the skew
    rows, cols = image.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    corrected_image = cv.warpAffine(image, rotation_matrix, (cols, rows), flags=cv.INTER_LINEAR)

    return corrected_image

def is_text_based(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                return text
            else:
                return False
    except Exception as e:
        print("Error:", e)
        return False

def deskew(docs):
    docs_desk = []
    for doc in docs:
        doc = correct_skew(doc)
        docs_desk.append(doc)
    return docs_desk

def filter_docs(docs):
    docs_filtered = []
    for doc in docs:
        filtered = filter_image(doc)
        multi = np.stack([filtered] * 3, axis=-1)
        docs_filtered.append(multi)
    return docs_filtered

def ocr(docs, deskew_flag =False, filter_flag =False):
    from doctr.models import ocr_predictor

    if filter_flag:
        docs = filter_docs(docs)
    if deskew_flag:
        docs = deskew(docs)
    
    model = ocr_predictor(det_arch='db_mobilenet_v3_large',
                          reco_arch='crnn_mobilenet_v3_large',
                          assume_straight_pages= True,
                          pretrained=True,
                          export_as_straight_boxes= True).cuda()
    results = model(docs)
    return results

def hocr(path, deskew_flag =False, filter_flag =False):
    from pikepdf import Pdf
    from PIL.Image import fromarray
    from doctr.io import DocumentFile

    docs = DocumentFile.from_pdf(path, scale = 5)
    result = ocr(docs, deskew_flag, filter_flag)

    xml_outputs = result.export_as_xml()

    pdf_output = Pdf.new()

    for (xml, img) in zip(xml_outputs, docs):
        
        hocr = HocrTransform(
            hocr= xml[1],
            dpi=300
        )

        pdf = hocr.to_pdf(image=fromarray(img))
        
        pdf_output.pages.extend(pdf.pages)
    
    path_noex = path.rsplit('.', 1)[0]
    path_hOCR = f'{path_noex}_hOCR.pdf'
    pdf_output.save(path_hOCR)

    return path_hOCR#, pdf_output

def pdf_to_text(path):
    text = is_text_based(path)
    
    if text:
        pass
    else:
        path_hocr = hocr(path)
        with fitz.open(path_hocr) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
    
    return text

def count_tokens(text: str) -> int:
    encoding = get_encoding('cl100k_base')
    return len(encoding.encode(text))

def chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 128,
    chunk_overlap  = 24,
    length_function = count_tokens,
    )

    chunks = text_splitter.create_documents([text])

    return chunks

def main(path):
    text = pdf_to_text(path)
    chunks = chunk(text)
    return chunks
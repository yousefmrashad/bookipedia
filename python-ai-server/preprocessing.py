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
    """
    Applies a series of image filtering and thresholding operations to enhance the image.

    Args:
        img (np.ndarray | None): The input image as a NumPy array.
        kSize (int): The size of the kernel for the filter. Default is 55.
        whitePoint (int): The threshold value for white points. Default is 120.
        blackPoint (int): The threshold value for black points. Default is 70.

    Returns:
        np.ndarray: The filtered and enhanced image as a NumPy array.
    """
    
    if kSize % 2 == 0:
        kSize += 1
        
    # Create a kernel for filtering
    kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
    filtered = cv.filter2D(img, -1, kernel)
    
    # Enhance the image by subtracting the filtered image and adding a constant value
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    
    # Apply thresholding to obtain a binary image
    _, img = cv.threshold(filtered, whitePoint, 255, cv.THRESH_TRUNC)
    
    # Map pixel values to a new range
    img = map_values(img.astype('int32'), 0, whitePoint, 0, 255).astype('uint8')
    
    img = map_values(img.astype('int32'), blackPoint, 255, 0, 255)
    
    # Apply thresholding to remove low intensity pixels
    _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
    img = img.astype('uint8')

    # Convert image to LAB color space and perform color manipulation
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    (l, a, b) = cv.split(lab)
    img = cv.add(cv.subtract(l, b), cv.subtract(l, a))
    
    # Apply sharpening filter
    sharpened = cv.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        
    return sharpened

def correct_skew(image):
    """
    Corrects the skew of an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The corrected image.
    """
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
    """
    Check if a PDF file contains text-based content.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str or bool: If the PDF file contains text-based content, the function returns the extracted text as a string.
                     If the PDF file does not contain any text-based content, the function returns False.
    """
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
        filtered = filter_image(doc)  # Apply image filtering
        multi = np.stack([filtered] * 3, axis=-1)  # Convert filtered image to RGB format
        docs_filtered.append(multi)  # Add filtered image to the list
    return docs_filtered

def ocr(docs, deskew_flag=False, filter_flag=False):
    """
    Perform OCR (Optical Character Recognition) on a list of documents.

    Args:
        docs (list): List of document images to perform OCR on.
        deskew_flag (bool, optional): Flag indicating whether to deskew the documents before OCR. Defaults to False.
        filter_flag (bool, optional): Flag indicating whether to filter the documents before OCR. Defaults to False.

    Returns:
        doctr Document: The OCR results as a Document object.
    """
    from doctr.models import ocr_predictor

    # Apply image filtering if filter_flag is True
    if filter_flag:
        docs = filter_docs(docs)

    # Deskew the documents if deskew_flag is True
    if deskew_flag:
        docs = deskew(docs)

    # Initialize the OCR model
    model = ocr_predictor(det_arch='db_mobilenet_v3_large',
                          reco_arch='crnn_mobilenet_v3_large',
                          assume_straight_pages=True,
                          pretrained=True,
                          export_as_straight_boxes=True).cuda()

    # Perform OCR on the documents
    results = model(docs)

    return results

def hocr(path, deskew_flag=False, filter_flag=False):
    """
    Perform OCR on a PDF document and save the result as a PDF with hOCR format.

    Args:
        path (str): The path to the input PDF file.
        deskew_flag (bool, optional): Flag indicating whether to deskew the input images. Defaults to False.
        filter_flag (bool, optional): Flag indicating whether to apply image filters. Defaults to False.

    Returns:
        str: The path to the output PDF file with hOCR format.
    """
    from pikepdf import Pdf
    from PIL.Image import fromarray
    from doctr.io import DocumentFile

    # Load the PDF document as a DocumentFile
    docs = DocumentFile.from_pdf(path, scale=5)
    
    # Perform OCR on the document
    result = ocr(docs, deskew_flag, filter_flag)

    # Export OCR results as XML
    xml_outputs = result.export_as_xml()

    # Create a new PDF document
    pdf_output = Pdf.new()

    # Iterate over each page in the OCR results
    for (xml, img) in zip(xml_outputs, docs):
        
        # Create an HocrTransform object
        hocr = HocrTransform(
            hocr=xml[1],
            dpi=300
        )

        # Convert the HOCR and image to PDF
        pdf = hocr.to_pdf(image=fromarray(img))
        
        # Append the PDF pages to the output document
        pdf_output.pages.extend(pdf.pages)
    
    # Generate the output file path
    path_noex = path.rsplit('.', 1)[0]
    path_hOCR = f'{path_noex}_hOCR.pdf'
    
    # Save the output PDF with hOCR format
    pdf_output.save(path_hOCR)

    return result.export()

def json_to_text(json):
    # Extracting text from "value" keys
    lines = []
    for page in json["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                text_values = []
                for word in line["words"]:
                    text_values.append(word["value"])
                lines.append(text_values)
                
    # Writing text values to a text file
    text = ''
    for line in lines:
        for value in line:
            text += value + ' '
        text += '\n'
    return text

def pdf_to_text(path):
    """
    Convert a PDF file to text.

    Parameters:
    path (str): The path to the PDF file.

    Returns:
    str: The extracted text from the PDF file.
    """
    # Check if the PDF is already text-based
    text = is_text_based(path)
    
    if text:
        pass
    else:
        # Perform OCR on the PDF
        json = hocr(path)
        # Convert the OCR results to text
        text = json_to_text(json)
    return text

def count_tokens(text: str) -> int:
    encoding = get_encoding('cl100k_base')
    return len(encoding.encode(text))

def chunk(text):
    """
    Splits the given text into smaller chunks.

    Args:
        text (str): The text to be chunked.

    Returns:
        langchain Document: The chunked text as a Document object.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=24,
        length_function=count_tokens,
    )

    chunks = text_splitter.create_documents([text])

    return chunks

def process(path):
    """
    Preprocesses a PDF file and returns the text chunks.

    Args:
        path (str): The path to the PDF file.

    Returns:
        list: A list of text chunks extracted from the PDF.
    """
    text = pdf_to_text(path)
    chunks = chunk(text)
    return chunks
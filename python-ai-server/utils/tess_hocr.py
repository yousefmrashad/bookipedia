# Imports
from PyPDF2 import PdfMerger, PdfReader
import sys
import time
import pytesseract
from pdf2image import convert_from_path
import os
import io
from deskew import determine_skew
import numpy as np
from skimage.io import imread
from skimage.transform import rotate
from skimage.color import rgb2gray

def deskew(img):
    """
    Deskew the images using the deskew library.
    """
    img = img[:, :, :3]
    gray = rgb2gray(img)
    angle = determine_skew(gray)
    rotated = rotate(img, angle, resize= True) * 255
    img_desk = rotated.astype(np.uint8)
    return img_desk

def main():
    """
    Entry point of the program.
    
    This function takes two command-line arguments: [file path], [file type], [deskew].
    It processes the specified document based on the provided file type and generates an hOCR PDF file.
    
    """
    
    start_time = time.time()

    # Check if two arguments are provided

    if len(sys.argv) < 3:
        print("Usage: python tess_hocr.py [file path] [file type]: {img, im_dir, pdf} [deskew(optional)]: {deskew, None}")
        return
    
    if sys.argv[1] == "help" or sys.argv[1] == "-h":
        print("Usage: python tess_hocr.py [file path] [file type]: {img, im_dir, pdf} [deskew(optional)]: {deskew, None}")
        return
    

    # Extract arguments
    doc = sys.argv[1]

    doc_noex = doc.rsplit('.', 1)[0]

    type = sys.argv[2]

    if len(sys.argv) == 4: 
        desk = sys.argv[3]
        if desk == "deskew":
            desk = True
    else:
        desk = False

    if type == "img":
        # Load the image
        docs = [imread(doc)]

    elif type == "im_dir":
        # Get the list of image files in the doc directory
        image_files = [os.path.join('doc', file) for file in os.listdir('doc') if file.endswith(('.jpg', '.jpeg', '.png'))]

        # Load the images
        docs = [imread(file) for file in image_files]

    elif type == "pdf":
        docs = convert_from_path(doc)

    merger = PdfMerger()

    for img in docs:

        if desk:
            # Deskew the images
            img = deskew(img)
        
        pdf = pytesseract.image_to_pdf_or_hocr(img, lang='eng', extension='pdf')
        pdf_bytes_stream = io.BytesIO(pdf)
        pdf = PdfReader(pdf_bytes_stream)
        merger.append(pdf)
    merger.write(f'{doc_noex}_hOCR.pdf')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
if __name__ == "__main__":
    main()

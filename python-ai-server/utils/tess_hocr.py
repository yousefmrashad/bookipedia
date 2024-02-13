# Imports
from PyPDF2 import PdfMerger, PdfReader
import time
import pytesseract
from pdf2image import convert_from_path
import os
import io
from deskew import correct_skew
from skimage.io import imread
from filter import filter_image
import argparse


def main():
    """
    Entry point of the program.

    Usage: tess_hocr.py [-h] [--deskew] [--filter] doc {img,im_dir,pdf}
    It processes the specified document based on the provided file type and generates an hOCR PDF file.
    
    """
    
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process document and generate hOCR PDF file.')
    parser.add_argument('doc', type=str, help='Path to the document file')
    parser.add_argument('type', choices=['img', 'im_dir', 'pdf'], help='Type of the document: img, im_dir, or pdf')
    parser.add_argument("-d", "--deskew", action="store_true", help="Perform deskewing (optional)")
    parser.add_argument("-f", "--filter", action="store_true", help="Apply filter to the images (optional)")
    args = parser.parse_args()

    doc = args.doc
    doc_noex = doc.rsplit('.', 1)[0]
    type = args.type
    desk = args.deskew
    filter = args.filter

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
            img = correct_skew(img)
        if filter:
            # Apply filter
            img = filter_image(img)
        
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

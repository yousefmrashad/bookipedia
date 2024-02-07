# Imports
from tempfile import TemporaryDirectory
from PyPDF2 import PdfMerger
from PIL import Image
import sys
import time
import pytesseract
from pdf2image import convert_from_path
import os

def main():
    start_time = time.time()

    # Check if two arguments are provided

    if len(sys.argv) != 3:
        print("Usage: python tess_hocr.py [file path] [file type]: {img, im_dir, pdf}")
        return
    
    if sys.argv[1] == "help" or sys.argv[1] == "-h":
        print("Usage: python tess_hocr.py [file path] [file type]: {img, im_dir, pdf}")
        return
    

    # Extract arguments
    doc = sys.argv[1]

    doc_noex = doc.rsplit('.', 1)[0]

    type = sys.argv[2]

    if type == "img":
        # Load the image
        docs = [Image.open(doc)]

    elif type == "im_dir":
        # Get the list of image files in the doc directory
        image_files = [os.path.join('doc', file) for file in os.listdir('doc') if file.endswith(('.jpg', '.jpeg', '.png'))]

        # Load the images
        docs = [Image.open(file) for file in image_files]

    elif type == "pdf":
        docs = convert_from_path(doc)

    merger = PdfMerger()

    for i, img in enumerate(docs):
        with TemporaryDirectory() as tmpdir:

            # use pytesseract to extract text from the image
            pdf = pytesseract.image_to_pdf_or_hocr(img, lang='eng', extension='pdf')

            # print the extracted text
            with open(f'{tmpdir}/{i}.pdf', 'wb') as f:
                f.write(pdf)

            merger.append(f'{tmpdir}/{i}.pdf')

    merger.write(f'{doc_noex}_hOCR.pdf')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
if __name__ == "__main__":
    main()

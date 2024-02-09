# Imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL.Image import fromarray
from hocr import HocrTransform
import time
import pikepdf
from deskew import determine_skew
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray
from filter import filter_image
import argparse

def deskew(docs):
    """
    Deskew the images using the deskew library.
    """
    docs_desk = []
    for doc in docs:
        gray = rgb2gray(doc)
        angle = determine_skew(gray)
        rotated = rotate(doc, angle, resize= True) * 255
        docs_desk.append(rotated.astype(np.uint8))
    return docs_desk

def filter_docs(docs):
    """
    Apply a filter to the images.
    """
    docs_filtered = []
    for doc in docs:
        filtered = filter_image(doc)
        multi = np.stack([filtered] * 3, axis=-1)
        docs_filtered.append(multi)
    return docs_filtered

def main():
    """
    Entry point of the script.

    Usage: doctr_hocr.py [-h] [--deskew] [--filter] file_path {img,pdf}
    Extracts text from images or PDFs using OCR and saves the result as an hOCR PDF file.
    """

    start_time = time.time()

    # Check if two arguments are provided

    parser = argparse.ArgumentParser(description="Extracts text from images or PDFs using OCR and saves the result as an hOCR PDF file.")
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument("file_type", choices=["img", "pdf"], help="Type of the file: img or pdf")
    parser.add_argument("-d", "--deskew", action="store_true", help="Perform deskewing (optional)")
    parser.add_argument("-f", "--filter", action="store_true", help="Apply filter to the images (optional)")
    
    args = parser.parse_args()

    doc = args.file_path
    doc_noex = doc.rsplit('.', 1)[0]
    type = args.file_type
    desk = args.deskew
    filter = args.filter

    
    if type == "img":
        docs = DocumentFile.from_images(doc)
    elif type == "pdf":
        docs = DocumentFile.from_pdf(doc, scale = 4.1667)
    
    if desk:
        # Deskew the images
        docs = deskew(docs)
    if filter:
        # Apply a filter to the images
        docs = filter_docs(docs)
    
    model = ocr_predictor(det_arch='db_mobilenet_v3_large',
                          reco_arch='crnn_mobilenet_v3_large',
                          assume_straight_pages= True,
                          pretrained=True,
                          export_as_straight_boxes= True).cuda()

    result = model(docs)

    xml_outputs = result.export_as_xml()

    pdf_output = pikepdf.Pdf.new()

    for (xml, img) in zip(xml_outputs, docs):
        
        hocr = HocrTransform(
            hocr= xml[1],
            dpi=300
        )

        pdf = hocr.to_pdf(image=fromarray(img))
        
        pdf_output.pages.extend(pdf.pages)

    pdf_output.save(f'{doc_noex}_hOCR.pdf')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
if __name__ == "__main__":
    main()

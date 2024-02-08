# Imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL.Image import fromarray
from hocr import HocrTransform
import sys
import time
import pikepdf
from deskew import determine_skew
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray

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

def main():
    """
    Entry point of the script.
    Extracts text from images or PDFs using OCR and saves the result as an hOCR PDF file.

    Usage: python doctr_hocr.py [file path] [file type]: {img, pdf} [deskew(optional)]: {deskew, None}

    """

    start_time = time.time()

    # Check if two arguments are provided

    if len(sys.argv) < 3:
        print("Usage: python doctr_hocr.py [file path] [file type]: {img, pdf} [deskew(optional)]: {deskew, None}")
        return
    
    if sys.argv[1] == "help" or sys.argv[1] == "-h":
        print("Usage: python doctr_hocr.py [file path] [file type]: {img, pdf} [deskew(optional)]: {deskew, None}")
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
        docs = DocumentFile.from_images(doc)
    elif type == "pdf":
        docs = DocumentFile.from_pdf(doc, scale = 4.1667)
    
    if desk:
        # Deskew the images
        docs = deskew(docs)

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

# Imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
from hocr import HocrTransform
import sys
import time
import pikepdf

def main():
    """
    Entry point of the script.
    
    This function takes two command-line arguments: [file path] and [file type].
    The [file path] should be the path to the input document file, and the [file type]
    should be either 'img' for image files or 'pdf' for PDF files.
    
    The function performs OCR (Optical Character Recognition) on the input document
    using the specified file type. It uses the 'ocr_predictor' model to extract text
    from the document and generates an hOCR (HTML OCR) output. The hOCR output is then
    converted to a PDF file and saved with the same name as the input document, but
    with '_hOCR.pdf' appended to the filename.
    
    The execution time of the script is also printed at the end.
    """
    start_time = time.time()

    # Rest of the code...

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
def main():
    start_time = time.time()

    # Check if two arguments are provided

    if len(sys.argv) != 3:
        print("Usage: python doctr_hocr.py [file path] [file type]: {img, pdf}")
        return
    
    if sys.argv[1] == "help" or sys.argv[1] == "-h":
        print("Usage: python doctr_hocr.py [file path] [file type]: {img, pdf}")
        return
    

    # Extract arguments
    doc = sys.argv[1]

    doc_noex = doc.rsplit('.', 1)[0]

    type = sys.argv[2]

    if type == "img":
        docs = DocumentFile.from_images(doc)
    elif type == "pdf":
        docs = DocumentFile.from_pdf(doc, scale = 4.1667)

    model = ocr_predictor(det_arch='db_resnet50_rotation',
                          reco_arch='crnn_mobilenet_v3_large',
                          assume_straight_pages= False,
                          pretrained=True,
                          export_as_straight_boxes= True).cuda()

    result = model(docs)

    xml_outputs = result.export_as_xml()

    pdf_output = pikepdf.Pdf.new()

    for (xml, img) in enumerate(zip(xml_outputs, docs)):
        
        hocr = HocrTransform(
            hocr= xml[1],
            dpi=300
        )

        pdf = hocr.to_pdf(image=Image.fromarray(img))
        
        pdf_output.pages.extend(pdf.pages)

    pdf_output.save(f'{doc}_hOCR.pdf')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
if __name__ == "__main__":
    main()

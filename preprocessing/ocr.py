# Utils
from root_config import *
from utils.init import *

from utils.hocr import HocrTransform
# ===================================================================== #

# OCR
class OCR:
    def __init__(self, doc_path: str, scale=5):
        self.doc_path = doc_path
        self.scale = scale

        from doctr.io import DocumentFile
        self.docs = DocumentFile.from_pdf(doc_path, scale=scale)
    # ---------------------------------------------- #

    @staticmethod
    def filter_image(img: np.ndarray, kernel_size=55, white_point=120, black_point=70) -> np.ndarray:
        
        if (kernel_size % 2 == 0): kernel_size += 1
            
        # Create a kernel for filtering
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        filtered = cv.filter2D(img, -1, kernel)
        
        # Enhance the image by subtracting the filtered image and adding a constant value
        filtered = img.astype(np.float32) - filtered.astype(np.float32)
        filtered = filtered + 127 * np.ones(img.shape, np.uint8)
        filtered = filtered.astype(np.uint8)
        
        # Apply thresholding to obtain a binary image
        _, img = cv.threshold(filtered, white_point, 255, cv.THRESH_TRUNC)
        
        # Map pixel values to a new range
        img = map_values(img.astype(np.int32), 0, white_point, 0, 255).astype(np.uint8)
        img = map_values(img.astype(np.int32), black_point, 255, 0, 255)
        
        # Apply thresholding to remove low intensity pixels
        _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
        img = img.astype(np.uint8)

        # Convert image to LAB color space and perform color manipulation
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        (l, a, b) = cv.split(lab)
        img = cv.add(cv.subtract(l, b), cv.subtract(l, a))
        
        # Apply sharpening filter
        sharpened = cv.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        
        # Convert image to RGB format
        rgb_image = np.stack([sharpened] * 3, axis=-1)

        return rgb_image
    # ---------------------------------------------- #

    @staticmethod
    def correct_skew(img: np.ndarray) -> np.ndarray:

        # Convert image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
        rows, cols = img.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        corrected_image = cv.warpAffine(img, rotation_matrix, (cols, rows), flags=cv.INTER_LINEAR)

        return corrected_image
    # ---------------------------------------------- #

    # OCR
    def ocr(self) -> Document:
        from doctr.models import ocr_predictor
        model = ocr_predictor(det_arch=DETECTION_MODEL, reco_arch=RECOGNITION_MODEL,
                              assume_straight_pages=True, pretrained=True,
                              export_as_straight_boxes=True).cuda()
        
        self.result = model.forward(self.docs)
    # ---------------------------------------------- #

    # HOCR
    def hocr(self):
        # Export OCR results as XML
        xml_outputs = self.result.export_as_xml()

        # Create a new PDF document
        self.pdf_output = Pdf.new()

        # Iterate over each page in the OCR results
        for (xml, img) in zip(xml_outputs, self.docs):
            
            # Create an HocrTransform object
            hocr = HocrTransform(hocr=xml[1], dpi=360)

            # Convert the HOCR and image to PDF
            pdf = hocr.to_pdf(image = PIL.Image.fromarray(img), invisible_text=True)
            
            # Append the PDF pages to the output document
            self.pdf_output.pages.extend(pdf.pages)
    # ---------------------------------------------- #

    # Save the output PDF with HOCR format
    def save_hocr_doc(self) -> None:
        # hocr_doc_path = self.doc_path.replace(".pdf", "_hocr.pdf")
        self.pdf_output.save(self.doc_path)
    # ---------------------------------------------- #

    # OCR Pipeline
    def apply_ocr(self, deskew=False, filter=False) -> None:
        if (filter):
            self.docs = [OCR.filter_image(page) for page in self.docs]
        if (deskew):
            self.docs = [OCR.correct_skew(page) for page in self.docs]

        self.ocr()
        self.hocr()
        self.save_hocr_doc()
    # ---------------------------------------------- #
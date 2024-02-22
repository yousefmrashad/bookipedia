from utils_config import *
from utils import *
from doctr.models import ocr_predictor
from doctr.io import DocumentFile, Document
# -------------------------------------------------------------------- #

# HOCR
class HocrParser:

    def __init__(self):
        self.box_pattern = re.compile(r'bbox((\s+\d+){4})')
        self.baseline_pattern = re.compile(r'baseline((\s+[\d\.\-]+){2})')

    def _element_coordinates(self, element: Element) -> Dict:
        """
        Returns a tuple containing the coordinates of the bounding box around
        an element
        """
        out = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        if ('title' in element.attrib):
            matches = self.box_pattern.search(element.attrib['title'])
            if (matches):
                coords = matches.group(1).split()
                out = {'x1': int(coords[0]), 'y1': int(coords[1]), 'x2': int(coords[2]), 'y2': int(coords[3])}
        return out

    def _get_baseline(self, element: Element) -> Tuple[float, float]:
        """
        Returns a tuple containing the baseline slope and intercept.
        """
        if ('title' in element.attrib):
            matches = self.baseline_pattern.search(element.attrib['title']).group(1).split()
            if (matches):
                return float(matches[0]), float(matches[1])
        return (0.0, 0.0)

    def _pt_from_pixel(self, pxl: Dict, dpi: int) -> Dict:
        """
        Returns the quantity in PDF units (pt) given quantity in pixels
        """
        pt = [(c / dpi * inch) for c in pxl.values()]
        return {'x1': pt[0], 'y1': pt[1], 'x2': pt[2], 'y2': pt[3]}

    def _get_element_text(self, element: Element) -> str:
        """
        Return the textual content of the element and its children
        """
        text = ''
        if (element.text is not None):
            text += element.text
        for child in element:
            text += self._get_element_text(child)
        if (element.tail is not None):
            text += element.tail
        return text

    def export_pdfa(self,
                    out_filename: str,
                    hocr: ET.ElementTree,
                    image: Optional[np.ndarray] = None,
                    fontname: str = "Times-Roman",
                    fontsize: int = 1,
                    invisible_text: bool = True,
                    add_spaces: bool = False,
                    dpi: int = 360):
        """
        Generates a PDF/A document from a hOCR document.
        """

        width, height = None, None
        # Get the image dimensions
        for div in hocr.findall(".//div[@class='ocr_page']"):
            coords = self._element_coordinates(div)
            pt_coords = self._pt_from_pixel(coords, dpi)
            width, height = pt_coords['x2'] - pt_coords['x1'], pt_coords['y2'] - pt_coords['y1']
            # after catch break loop
            break
        if (width is None or height is None):
            raise ValueError("Could not determine page size")

        pdf = Canvas(out_filename, pagesize=(width, height), pageCompression=1)

        span_elements = [element for element in hocr.iterfind(".//span")]
        for line in span_elements:
            if ('class' in line.attrib) and (line.attrib['class'] == 'ocr_line') and (line is not None):
                # get information from xml
                pxl_line_coords = self._element_coordinates(line)
                line_box = self._pt_from_pixel(pxl_line_coords, dpi)

                # compute baseline
                slope, pxl_intercept = self._get_baseline(line)
                if (abs(slope) < 0.005):
                    slope = 0.0
                angle = atan(slope)
                cos_a, sin_a = cos(angle), sin(angle)
                intercept = pxl_intercept / dpi * inch
                baseline_y2 = height - (line_box['y2'] + intercept)

                # configure options
                text = pdf.beginText()
                text.setFont(fontname, fontsize)
                pdf.setFillColor(black)
                if (invisible_text):
                    text.setTextRenderMode(3)  # invisible text

                # transform overlayed text
                text.setTextTransform(cos_a, -sin_a, sin_a, cos_a, line_box['x1'], baseline_y2)

                elements = line.findall(".//span[@class='ocrx_word']")
                for elem in elements:
                    elemtxt = self._get_element_text(elem).strip()
                    # replace unsupported characters
                    elemtxt = elemtxt.translate(str.maketrans({'ﬀ': 'ff', 'ﬃ': 'f‌f‌i', 'ﬄ': 'f‌f‌l', 'ﬁ': 'fi', 'ﬂ': 'fl'}))
                    if (not elemtxt):
                        continue

                    # compute string width
                    pxl_coords = self._element_coordinates(elem)
                    box = self._pt_from_pixel(pxl_coords, dpi)
                    if (add_spaces):
                        elemtxt += ' '
                        box_width = box['x2'] + pdf.stringWidth(elemtxt, fontname, fontsize) - box['x1']
                    else:
                        box_width = box['x2'] - box['x1']
                    font_width = pdf.stringWidth(elemtxt, fontname, fontsize)

                    # Adjust relative position of cursor
                    cursor = text.getStartOfLine()
                    dx = box['x1'] - cursor[0]
                    dy = baseline_y2 - cursor[1]
                    text.moveCursor(dx, dy)

                    # suppress text if it is 0 units wide
                    if (font_width > 0):
                        text.setHorizScale(100 * box_width / font_width)
                        text.textOut(elemtxt)
                pdf.drawText(text)

        # overlay image if provided
        if (image is not None):
            pdf.drawImage(ImageReader(PIL.Image.fromarray(image)),
                          0, 0, width=width, height=height)
        pdf.save()
# -------------------------------------------------------------------- #

# OCR
class OCR:
    def __init__(self, doc_path: str, scale: int = 5):
        self.doc_path = doc_path
        self.scale = scale
        self.docs = DocumentFile.from_pdf(doc_path, scale=scale)
    # -------------------------------------------------------------------- #

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
    # -------------------------------------------------------------------- #

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
    # -------------------------------------------------------------------- #

    # OCR
    def ocr(self) -> Document:
        model = ocr_predictor(det_arch=DETECTION_MODEL,
                            reco_arch=RECOGNITION_MODEL,
                            assume_straight_pages=True,
                            pretrained=True,
                            export_as_straight_boxes=True).cuda()
        
        result = model.forward(self.docs)
        self.result: Document = result
    # -------------------------------------------------------------------- #

    # HOCR
    def hocr(self):
        xml_output = self.result.export_as_xml()

        parser = HocrParser()
        pdf_merger = PyPDF2.PdfMerger()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (xml, img) in enumerate(zip(xml_output, self.docs)):
                page_tmpdir = os.path.join(tmpdir, f"{i+1}.pdf")
                parser.export_pdfa(page_tmpdir, hocr=xml[1], image=img)
                pdf_merger.append(page_tmpdir)
            
            pdf_merger.write(self.doc_path)
            pdf_merger.close()
    # -------------------------------------------------------------------- #

    # OCR Pipeline
    def apply_ocr(self, handwritten=False):
        if (handwritten):
            self.docs = list(map(lambda doc: OCR.correct_skew(OCR.filter_image(doc)), self.docs))

        self.ocr()
        self.hocr()
# -------------------------------------------------------------------- #
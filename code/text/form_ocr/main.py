import cv2
import matplotlib.pyplot as plt
from align_images import align_images
from collections import namedtuple
import pytesseract

base_path = './text/form_ocr'

# Load Template and Image
image = cv2.imread(base_path + '/assets/scan_01.jpg')
template = cv2.imread(base_path + '/assets/template.png')


# Align Image with Template
aligned_image = align_images(image, template)


# Define sections for OCR
OCRSection = namedtuple('OCRSection', ['id', 'bbox', 'filters'])
ocrSections = [
    OCRSection('first_name', [(265, 240), (1010, 335)], ['first', 'name', 'middle', 'initial']),
    OCRSection('last_name', [(1015, 235), (1855, 335)], ['last', 'name']),
    OCRSection('address', [(265, 340), (1855, 435)], ['address']),
    OCRSection('address', [(265, 440), (1855, 535)], ['city', 'town', 'state', 'zip code']),
    OCRSection('employer_address', [(245, 2712), (1510, 2910)], ['employer', 'name', 'address'])
]

# Utility method for filtering the OCR section headers
def filter_text(text, filters):
    final_text = []
    lines = text.split('\n')
    for line in lines:
        words = [word for word in filters if word in line.lower()]
        if len(words) == 0:
            line = ''.join(char for char in line if ord(char) < 128)
            final_text.append(line)
    return '\n'.join(final_text) 

# Iterate over the OCR Sections and OCR the image ROIs
for ocrSection in ocrSections:
    ((x1, y1), (x2, y2)) = ocrSection.bbox
    roi = aligned_image[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    text = filter_text(text, ocrSection.filters)
    
    # Label Text
    cv2.rectangle(aligned_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(aligned_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

cv2.imshow('Form', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
from easyocr import Reader
import cv2

base_path = './text/easy_ocr'
image = cv2.imread(base_path + '/assets/car_wash.png')

reader = Reader(['en'])
result = reader.readtext(image)

for (bbox, text, score) in result:
    
    tl = tuple(map(int, bbox[0]))
    br = tuple(map(int, bbox[2]))
    
    cv2.rectangle(image, tl, br, (0, 255, 0), 3)
    cv2.putText(image, text, (tl[0], tl[1]+75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Image', image)    
cv2.waitKey(0)
cv2.destroyAllWindows()
    
     
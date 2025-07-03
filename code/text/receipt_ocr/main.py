import cv2
import imutils
from imutils.perspective import four_point_transform
import pytesseract

base_path = './text/receipt_ocr'
image = cv2.imread(base_path + '/assets/whole_foods.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,3), 0)

edged = cv2.Canny(blurred, 50, 100)

cntrs = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = imutils.grab_contours(cntrs)
c = max(cntrs, key=cv2.contourArea)


peri = cv2.arcLength(c, closed=True)
approx = cv2.approxPolyDP(c, 0.02*peri, closed=True)

if len(approx) != 4:
    print('Not able to find the receipt region ', len(approx))
    exit()

# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the receipt
receipt = four_point_transform(image, approx.reshape(4, 2))

#cv2.imshow('Receipt', receipt)
#cv2.waitKey(0)


text = pytesseract.image_to_string(
    cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB),
    config="--psm 4")

print(text)




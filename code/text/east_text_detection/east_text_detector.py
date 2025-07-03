import cv2
import numpy as np
import pytesseract

# Path to the image and the EAST model
base_path = './text/east_text_detection'
image_path = base_path + '/images/car_wash.png'  # <-- Replace with your actual image path
east_model = base_path + '/frozen_east_text_detection.pb'

# Load image
image = cv2.imread(image_path)
orig = image.copy()
(h, w) = image.shape[:2]

# Set new width and height (must be multiple of 32)
(newW, newH) = (320, 320)
rW = w / float(newW)
rH = h / float(newH)

# Resize and prepare input blob
image = cv2.resize(image, (newW, newH))
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

# Load the pretrained EAST model
net = cv2.dnn.readNet(east_model)
net.setInput(blob)

# Forward pass
scores, geometry = net.forward([
    "feature_fusion/Conv_7/Sigmoid",     # Score map
    "feature_fusion/concat_3"            # Geometry map
])

# Decode predictions
def decode_predictions(scores, geometry, conf_threshold=0.5):
    (numRows, numCols) = scores.shape[2:4]
    boxes = []
    confidences = []



    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue

            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            endX = int(x * 4.0 + cos * x1[x] + sin * x2[x])
            endY = int(y * 4.0 - sin * x1[x] + cos * x2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    return (boxes, confidences)

# Decode and apply NMS
boxes, confidences = decode_predictions(scores, geometry)
rects = []
for (startX, startY, endX, endY) in boxes:
    rects.append([startX, startY, endX - startX, endY - startY])  # x, y, w, h

indices = cv2.dnn.NMSBoxes(rects, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw final boxes
for i in indices:
    i = i[0] if isinstance(i, (list, np.ndarray)) else i
    (x, y, w, h) = rects[i]
    startX = int(x * rW)
    startY = int(y * rH)
    endX = int((x + w) * rW)
    endY = int((y + h) * rH)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    roi = orig[startY:endY, startX:endX]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 7")
    print("Text: ",  text)
    cv2.putText(orig, text.strip(), (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

# Show result
cv2.imshow("EAST Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()

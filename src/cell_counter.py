import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PIL import Image

# THIS FUNCTION HAS BEEN ISOLATED FOR TESTING PURPOSES

image_path = "data/lung_image_sets/lung_aca/lungaca1368.jpeg"
# read the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    print("Error: Unable to load the image.")
    exit()
# convert the image to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# reduce noise by applying the gaussian blur function
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# equalized_hist = cv2.equalizeHist(blurred)
# cv2.imshow("eq hist", equalized_hist)
block_size = 43
constant_value = 7
# use adaptive thresholding to try and identify cells to classify each type of image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, constant_value)

# Apply morphological operations to remove small noise
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
morph = cv2.dilate(morph, kernel, iterations=1)

# find contours which could correspond to cell nuclei
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
# filter out small contours that are not cells
cells = []
area_total = []
for c in contours:
     area_total.append(cv2.contourArea(c))

for c in contours:
    area = cv2.contourArea(c)
    if area > (np.mean(area_total) - np.std(area_total)):
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if 0.5 < circularity < 1.5:  # range for near-circular shapes
                cells.append(c)
# show the cell count on the image
output_img = image.copy()
for c in cells:
    # Use bounding rectangles to draw an approximate region around each cell
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Img with bounding boxes", output_img)
print(len(cells))
cv2.waitKey(0)
cv2.destroyAllWindows()
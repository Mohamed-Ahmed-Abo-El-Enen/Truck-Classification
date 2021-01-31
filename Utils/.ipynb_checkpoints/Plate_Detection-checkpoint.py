import cv2 as cv
import numpy as np

def get_plates(image):
    # RGB to Gray scale conversion
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #cv.imshow("1 - Grayscale Conversion", gray)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    #cv.imshow("2 - Bilateral Filter", gray)

    # Find Edges of the grayscale image
    edged = cv.Canny(gray, 170, 200)
    #cv.imshow("4 - Canny Edges", edged)

    # Find contours based on Edges
    cnts, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]  
    NumberPlateCnt = None  # we currently have no Number plate contour

    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            break

    # Drawing the selected contour on the original image
    #cv.drawContours(image, NumberPlateCnt, -1, (0, 0, 0), 1)
    # cv.imshow("Final Image With Number Plate Detected", image)
    return NumberPlateCnt



def square(img):
    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image

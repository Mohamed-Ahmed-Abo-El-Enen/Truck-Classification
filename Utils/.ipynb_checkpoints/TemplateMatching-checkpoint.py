import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import os



def find_brand_template(image, template):
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    box = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)  
        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    box = ((startX, startY), (endX, endY))
    return box, maxVal


def get_brand_label(image, template_images_path):    
    max_match = 0
    best_template_box_match = None
    best_brand_match = None
    for template_path in template_images_path:
        seg = str(template_path).split('/')
        label = seg[-2]
        template = cv2.imread(template_path, 0)
        box, maxVal = find_brand_template(image, template)
        
        if maxVal > max_match:
            max_match = maxVal
            best_template_box_match = box
            best_brand_match = label

    if best_template_box_match is not None:
        (startX, startY), (endX, endY) = best_template_box_match
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, best_brand_match, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        
    return best_brand_match


def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".jpg"):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  

    return file_paths  



















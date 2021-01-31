import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob




def segment_plate(plate):
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_Abs = cv2.convertScaleAbs(plate, alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_Abs, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    return plate, binary, thre_mor


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def crop_characters(plate, binary, thre_mor):
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate.copy()

    # Initialize a list which will be used to append charater image
    cropped_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cropped_characters.append(curr_num)
                
    return cropped_characters


# pre-processing input images and pedict with model
def predict_from_model(image,model,labels, width=80, height=80):
    image = cv2.resize(image,(width, height))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


def recognize_plate_characters(plate, model,labels):
    fig = plt.figure(figsize=(15,3))
    plate, binary, thre_mor = segment_plate(plate)
    cropped_characters = crop_characters(plate, binary, thre_mor)
    cols = len(cropped_characters)
    final_string = ''
    grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

    for i,character in enumerate(cropped_characters):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character,model,labels))
        plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        plt.axis(False)
        plt.imshow(character,cmap='gray')

    return final_string





















from cv2 import cv2
import numpy as np
from tensorflow.keras.models import load_model


# READ THE MODEL WEIGHTS
def intializePredectionModel():
    model = load_model('Resources/myModel.h5')
    return model

# Preprocess (Gray->Blur->Binary->Dilate(if needed) for better edges)
def PreprocessImg(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Change to Gray as more accuracy for binary form or less colors.
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  # Blur the input img to reduce noises from unwanted sharp ones. Convolves img with specified Gaussian kernel
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255, 1, 1, 11, 2) # Gray to binary . Applies different thresholds to different points based on the mean
    return imgThreshold

# Find the biggest Contour assuring that it has max area with 4 corners(Sudoku)
def biggestContour(contours):
    biggest = np.array([]) # empty array to store the corner_points
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True) # Arclrngth/Perimeter(if closed) of the curve
            corner_points = cv2.approxPolyDP(i, 0.02 * peri, True) # Gives a approx curve with fewer points to get corner points of that object
            if area > max_area and len(corner_points) == 4:
                biggest = corner_points
                max_area = area
    return biggest, max_area

# Reorder the points in form to apply WARP_PERSPECTIVE on img
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # reducing one axis(as its unneccessary)
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32) # Empty array to store new points in shape of its original as its expected for further process
    add = myPoints.sum(axis=1) # Sum elements in a row(axis=1)
    myPointsNew[0] = myPoints[np.argmin(add)] # Row points with min sum
    myPointsNew[3] =myPoints[np.argmax(add)] # Row points with max sum
    diff = np.diff(myPoints, axis=1) # Subtract elements in a row (1st element - 2nd element)
    myPointsNew[1] =myPoints[np.argmin(diff)] # Row points with min diff
    myPointsNew[2] = myPoints[np.argmax(diff)] # Row points with max diff
    return myPointsNew


#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,9)  # Split rows  eg. [[1],[2],[3]] into [1],[2],[3]
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)  # Split elements in each row
        for box in cols:
            boxes.append(box)
    return boxes


#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## Prepare img as to feed in the model
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255 # Normalaization
        img = img.reshape(1, 28, 28, 1)

        ## GET PREDICTION
        classIndex = model.predict_classes(img) # returns [7]
        predictions = model.predict(img) # returns probabilities of all classes in a array
        probabilityValue = np.amax(predictions) # Max probability among that array (predictions)
        # All empty spaces have least probability of prediction with classIndex 0 . ALl detected digits have 90%+ probability .

        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

# Now, the result is sen to the main function


#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


# Stack images in single window in vertical/horizontal arrays
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

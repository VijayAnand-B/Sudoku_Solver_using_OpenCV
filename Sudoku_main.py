# TO REMOVE WARININGS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports
import Sudoku_solver
from utils import *
from cv2 import cv2
import numpy as np

########################
path = 'Resources/1.jpg'
widthImg = 450
heightImg = 450
model = intializePredectionModel()  # Load the CNN model
######################

# 1. PREPROCESSING (Gray->Blur->Binary->Dilate(if needed) for better edges)
img = cv2.imread(path)
img = cv2.resize(img,(widthImg,heightImg))  # Resize img to square shape
imgBlank = np.zeros((heightImg,widthImg,3), np.uint8 ) # Blank img(rgb format) used for testing
imgThreshold = PreprocessImg(img) # Function


# 2. FIND ALL CONTOURS
imgContour = img.copy() # img to plot all contours
imgBigContour = img.copy() # Img to plot biggest contour
contours, Hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
cv2.drawContours(imgContour, contours, -1,(0,255,0), 1)

# 3. FIND BIGGEST CONTOUR(sudoku box)  & USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours)
# print(biggest, biggest.dtype)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 15)

    # Apply WARP_PERSPECTIVE to focus Sudoku Box on img
    pts1 = np.float32(biggest)  #Define points to warp in this img
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # Define corners of ur output img according to the order given in pts1
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculates a perspective transform to convert from pts1 to pts2
    imgWarpColoured = cv2.warpPerspective(img, matrix,(widthImg, heightImg))  # Applies Perspective Transform on img to get destination img
    imgWarpColoured = cv2.cvtColor(imgWarpColoured,cv2.COLOR_BGR2GRAY) # Convert to Gray for Number Classification

    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgDetectedDigits = imgBlank.copy()
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColoured)

    numbers = getPredection(boxes, model)   # [2,1,0,0,6,....]

    # imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color=(255,0,255))

    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    #### 5. FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers, 9)
    print(board)

    try:
        Sudoku_solver.solve(board)
    except:
        pass
    print(board)

    # Convert array into [2,1,3,....] format
    flatList = []
    for rows in board:
        for element in rows:
            flatList.append(element)
    SolvedDigits = flatList * posArray
    print(SolvedDigits)
    imgSolvedDigits = displayNumbers(imgSolvedDigits, SolvedDigits, color=(0, 255, 0))



    ### 6. OVERLAY THE SOLUTION
    pts2 = np.float32(biggest)  #Define points to warp in this img
    pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # Define corners of ur output img according to the order given in pts1
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculates a perspective transform to convert from pts1 to pts2
    imginvWarpColoured = img.copy
    imginvWarpColoured = cv2.warpPerspective(imgSolvedDigits, matrix,(widthImg, heightImg))  # Applies Perspective Transform on img to get destination img
    # final_img = cv2.addWeighted(img,1,imginvWarpColoured,3,1)

    imginvWarpGray = cv2.cvtColor(imginvWarpColoured,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imginvWarpGray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in original img
    img_bg = cv2.bitwise_and(img, img, mask=mask_inv)

    # Put Solved_sol in original img
    final_img = cv2.add(img_bg, imginvWarpColoured)

    imgArray = ([img, imgThreshold, imgContour, imgBigContour], [imgWarpColoured, imginvWarpColoured, img_bg,final_img])
    imgStack = stackImages(1,imgArray)
    cv2.putText(imgStack, '1. Input img', (100, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '2. Preprocessing', (517, 19), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '3. Contour Detection', (995, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '4. Find Sudoku Box', (1445, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '5. Warp Sudoku', (100, 890), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '6. Find Solution', (550, 890), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '7. Mask on img', (1000, 890), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgStack, '8. Overlay solution', (1450, 890), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('Output',imgStack)
    cv2.imwrite('Output.jpg', imgStack)
    cv2.waitKey(0)

    # if cv2.waitKey(1) == ord('q'):
    #     break


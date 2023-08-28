import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

print("starting video")
cap = cv2.VideoCapture(0)
print("video started")
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

while True:
    try:
        success, img = cap.read()  # read() return a touple (bool, img)
        hands, img = detector.findHands(img) # findHands() return a touple (array, img)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox'] # bounding box
            imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
            palm = img[y-offset : y+h+offset,x-offset:x+w+offset]
            
            aspectRatio = h/w

            if (aspectRatio>1):
                k = imgSize/h
                newWidth = math.ceil(k*w)
                palmResize = cv2.resize(palm,(newWidth,imgSize))
                palmDimensions = palmResize.shape # shape return touple (height,width)
                gap = math.ceil((imgSize-newWidth)/2)
                imgWhite[:,gap:gap+palmDimensions[1]] = palmResize

            # if (aspectRatio>1):
            #     k = imgSize/w
                # newHeight = math.ceil(k*h)
                # palmResize = cv2.resize(palm,(imgSize,newHeight))
                # palmDimensions = palmResize.shape
                # gap = math.ceil((imgSize-newHeight)/2)
                # imgWhite[gap:gap+palmDimensions[0],:] = palmResize
            
            else:
                k = imgSize/w
                newHeight = math.ceil(k*h)
                palmResize = cv2.resize(palm,(imgSize,newHeight))
                palmDimensions = palmResize.shape # shape return touple (height,width)
                gap = math.ceil((imgSize-newHeight)/2)
                imgWhite[gap:gap+palmDimensions[0],:] = palmResize


            cv2.imshow("White", imgWhite)
            cv2.imshow("Palm", palm)
    except:
        print("ERROR -------------------------------------------------------")
    cv2.imshow("Cam", img)
    cv2.waitKey(1)
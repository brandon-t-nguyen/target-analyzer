#!/bin/python3

import sys
import math
import numpy as np
import cv2

state, p1x, p1y, p2x, p2y, distance = 0,0,0,0,0,0
def calculateDispersion(circles, pixelDist, realDist):
    n = 0
    meanX = 0
    meanY = 0
    for i in circles[0,:]:
        n += 1
        meanX += i[0]
        meanY += i[1]

    if n > 0:
        meanX = (meanX/n)
        meanY = (meanY/n)
        meanDist = 0
        for i in circles[0,:]:
            meanDist += math.sqrt(math.pow(i[0] - meanX, 2) + math.pow(i[1] - meanY, 2))
        meanDist /= n
        meanRealDist = meanDist * realDist / pixelDist
        #print("Conversion: %f pixels to %f in" % (pixelDist, realDist))
        print("Mean distance: %f in" % meanRealDist)
    

def mouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Mouse: (%d, %d)" %(x, y))
        global state, p1x, p1y, p2x, p2y
        global circles
        if state == 0:
            p1x = x
            p1y = y
            state = 1
        elif state == 1:
            p2x = x
            p2y = y
            state = 0
            distance = math.sqrt(math.pow(p1x - p2x, 2) + math.pow(p1y - p2y, 2))
            realDistance = float(input("Input real distance (in):: "))
            calculateDispersion(circles, distance, realDistance)

def main():
    image_path = "target0.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]


    # Create window with freedom of dimensions
    cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    
    # Resize window to specified dimensions
    # cv2.resizeWindow("output", 400, 300)

    # MOA calculator callback
    cv2.setMouseCallback('output',mouseCallback)

    # Load an color image in grayscale
    img = cv2.imread(image_path, 0)

    # preprocessing
    pproc = img
    #pproc = cv2.blur(pproc,(15, 15))
    pproc = cv2.GaussianBlur(pproc, (35, 35), 1.25, 1.25)
    pproc = cv2.medianBlur(pproc, 15)
    pproc = cv2.bilateralFilter(pproc, 35, 0.35, 0.35)

    cv2.imshow('preprocess', pproc)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    dp = 1.2
    minDist = 40
    param1 = 70
    param2 = 70 * 5/8
    minRadius = 20
    maxRadius = 80

    circles = cv2.HoughCircles(pproc,cv2.HOUGH_GRADIENT,dp,minDist,
                               param1=param1,param2=param2,
                               minRadius=minRadius,maxRadius=maxRadius)

    circles = np.uint16(np.around(circles))
    meanX = 0
    meanY = 0
    n = 0
    for i in circles[0,:]:
        n += 1
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),-2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('output',cimg)
        print('%d: (%d, %d)' %(n, meanX, meanY))

        meanX += i[0]
        meanY += i[1]

    if n > 0:
        meanX = np.uint16(meanX/n)
        meanY = np.uint16(meanY/n)
        cv2.circle(cimg,(meanX,meanY),100,(255,0,255),-2)
        print('lol: (%d, %d)' %(meanX, meanY))
        cv2.imshow('output',cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#!/bin/python3

import sys
import math
import numpy as np
import cv2

from hole import Hole
from hough import Hough

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

# returns processed image
def preprocess(image):
    output = image;
    
    #output = cv2.blur(output, (15, 15))
    output = cv2.GaussianBlur(output, (25, 25), 10, 10)
    output = cv2.medianBlur(output, 25)
    output = cv2.bilateralFilter(output, 25, 10, 10)
    
    return output

def main():
    image_path = "target0.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]


    # Create window with freedom of dimensions
    cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    
    # Resize window to specified dimensions
    # cv2.resizeWindow("output", 400, 300)

    # MOA calculator callback
    cv2.setMouseCallback('output',mouseCallback)

    # Load an color image in grayscale
    img = cv2.imread(image_path, 0)

    # preprocessing
    proc = preprocess(img)

    cv2.imshow('preprocess', proc)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    hough = Hough()
    hough.dp     = 1.25
    hough.canny *= 4/8
    hough.accum *= 6/8
    cv2.imshow('edges', hough.runCanny(proc))
    circles = hough.runHough(proc)

    meanX = 0
    meanY = 0
    n = 0
    for i in circles:
        n += 1
        # draw the outer circle
        cv2.circle(cimg,(i.x,i.y),i.r,(0,255,0),-2)
        
        # draw the center of the circle
        cv2.circle(cimg,(i.x,i.y),2,(0,0,255),3)
        
        cv2.imshow('output',cimg)
        print('%d: (%d, %d)' %(n, meanX, meanY))

        meanX += i.x
        meanY += i.y

    if n > 0:
        meanX = np.uint16(meanX/n)
        meanY = np.uint16(meanY/n)
        cv2.circle(cimg,(meanX,meanY),100,(255,0,255),-2)
        print('Mean: (%d, %d)' %(meanX, meanY))
        cv2.imshow('output',cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

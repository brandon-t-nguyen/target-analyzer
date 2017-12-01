#!/bin/python3

import sys
import math
import statistics
import numpy as np
import cv2

from hole import Hole
from hough import Hough

state, p1x, p1y, p2x, p2y, distance = 0,0,0,0,0,0
circles = []
selected = False
s0 = [0, 0]
s1 = [0, 0]
img = []
def calculateDispersion(circles, pixelDist, realDist):
    n = len(circles)

    if n > 0:
        meanX = statistics.mean(c.x for c in circles)
        meanY = statistics.mean(c.y for c in circles)
        meanDist = 0
        for i in circles:
            meanDist += math.sqrt(math.pow(float(i.x) - meanX, 2) + math.pow(float(i.y) - meanY, 2))
        meanDist /= n
        meanRealDist = meanDist * realDist / pixelDist
        #print("Conversion: %f pixels to %f in" % (pixelDist, realDist))
        print("Mean distance: %f in" % meanRealDist)

def originalCallback(event, x, y, flags, param):
    global selecting
    global s1x, s1y, s2x, s2y
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse down (%d, %d)" %(x, y))
        s0[0] = x
        s0[1] = y
    elif event == cv2.EVENT_LBUTTONUP:
        print("Mouse up (%d, %d)" %(x, y))
        s1[0] = x
        s1[1] = y
        # create the mask
        mask = np.zeros(img.shape, dtype="uint8")
        selected = True

def pprocCallback(event, x, y, flags, param):
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

def sharpen(image):
   kernel = np.array([
                      [1,4,6,4,1],
                      [4,16,24,16,4],
                      [6, 24, -476, 25, 6],
                      [4,16,24,16,4],
                      [1,4,6,4,1]
                     ])
   kernel = np.multiply(kernel, -1/256)
   output = cv2.filter2D(image, -1, kernel)
   return output

def m_erode(image, kernel, iterations = 1):
    return cv2.erode(image, kernel, iterations = iterations, borderType = cv2.BORDER_CONSTANT, borderValue = 0)

def m_dilate(image, kernel, iterations = 1):
    return cv2.dilate(image, kernel, iterations = iterations, borderType = cv2.BORDER_CONSTANT, borderValue = 0)

def m_hitmiss(image, kernel, iterations = 1):
    return cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel, iterations = iterations)

def m_top(image, kernel, iterations = 1):
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations = iterations)

def m_black(image, kernel, iterations = 1):
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations = iterations)

def m_close(image, kernel, iterations = 1):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = iterations)

def m_open(image, kernel, iterations = 1):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = iterations)

def morph_preprocess(image):
    output = image
    output = cv2.GaussianBlur(output, (3, 3), 1, 1)
    output = cv2.bilateralFilter(output, 3, 1, 1)

    # 11
    output = cv2.adaptiveThreshold(output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

    n = 1
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    output = m_close(output, k_rect, n)
    return output

def filter_preprocess(image):
    output = image
    #output = cv2.blur(output, (3, 3))
    output = cv2.GaussianBlur(output, (3, 3), 1, 1)
    #output = cv2.medianBlur(output, 3)
    output = cv2.bilateralFilter(output, 3, 1, 3)
    return output

# returns processed image
def preprocess(image):
    output = filter_preprocess(image)
    #output = morph_preprocess(image)
    return output

# returns a part of the image from the ROI corner coordinates
# return offset and transformation coordinates
def normalize_selection(image, s0, s1):
    SIZE = [256, 256] # w, h
    output = image[s0[1]:s1[1], s0[0]:s1[0]]

    fix_size = np.array(SIZE, dtype=np.float)
    roi_size = np.array([s1[0] - s0[0], s1[1] - s0[1]], dtype=np.float)

    # scale to a fixed size
    if np.prod(roi_size) > np.prod(fix_size):
        output = cv2.resize(output, (int(SIZE[0]), int(SIZE[1])), interpolation = cv2.INTER_AREA);
    else:
        output = cv2.resize(output, (int(SIZE[0]), int(SIZE[1])), interpolation = cv2.INTER_CUBIC);

    scale = fix_size / roi_size
    offset = s0

    print("Scale:")
    print(scale)
    print("Offset")
    print(offset)

    return output, scale, offset

# uses the scale and offset returns from normalize_selection
# to return circles in original coords
def transform_circles(circles, scale, offset):
    for c in circles:
        c.x = int(round(float(c.x) / scale[0] + offset[0]))
        c.y = int(round(float(c.y) / scale[1] + offset[1]))
        c.r = int(round(float(c.r) / ((scale[0]+scale[1])/2)))

def draw_circles(circles, cimg):
    meanX = 0
    meanY = 0
    n = 0
    for i in circles:
        n += 1
        # draw the outer circle
        cv2.circle(cimg,(i.x,i.y),i.r+5,(0,0,0),-2)
        cv2.circle(cimg,(i.x,i.y),i.r,(0,255,0),-2)

        # draw the center of the circle
        cv2.circle(cimg,(i.x,i.y),2,(0,0,255),3)

        cv2.imshow('output',cimg)
        print('%d: (%d, %d)' %(n, i.x, i.y))

        meanX += i.x
        meanY += i.y

    # draw the mean as a cross
    if n > 0:
        meanX = np.uint16(meanX/n)
        meanY = np.uint16(meanY/n)

        thicc = 5
        size  = 40
        # back
        p1 = (meanX - thicc * 2, meanY - size - thicc * 2)
        p2 = (meanX + thicc * 2, meanY + size + thicc * 2)
        cv2.rectangle(cimg, p1, p2, (0,0,0), -1)
        p1 = (meanX - size - thicc * 2, meanY - thicc * 2)
        p2 = (meanX + size + thicc * 2, meanY + thicc * 2)
        cv2.rectangle(cimg, p1, p2, (0,0,0), -1)

        # fore
        p1 = (meanX - thicc * 1, meanY - size - thicc * 1)
        p2 = (meanX + thicc * 1, meanY + size + thicc * 1)
        cv2.rectangle(cimg, p1, p2, (0,0,255), -1)
        p1 = (meanX - size - thicc * 1, meanY - thicc * 1)
        p2 = (meanX + size + thicc * 1, meanY + thicc * 1)
        cv2.rectangle(cimg, p1, p2, (0,0,255), -1)

        print('Mean: (%d, %d)' %(meanX, meanY))

def main():
    image_path = "target0.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]


    # Create window with freedom of dimensions
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # MOA calculator callback
    cv2.setMouseCallback('original', originalCallback)
    cv2.setMouseCallback('output', pprocCallback)

    # Load an color image in grayscale
    global img, mask, unselected
    img = cv2.imread(image_path, 0)

    cv2.imshow('original', img)

    print("Select ROI and hit any key")
    cv2.waitKey(0)

    # preprocessing
    global s0, s1
    sel, scale, offset = normalize_selection(img, s0, s1);
    proc = preprocess(sel)

    cv2.imshow('preprocess', proc)
    cv2.waitKey(0)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    hough = Hough()
    hough.dp        = 1.25
    hough.minDist   = 10
    hough.minRadius = 2
    hough.maxRadius = 10
    hough.canny     = 70
    hough.accum     = 20
    cv2.imshow('edges', hough.runCanny(proc))
    global circles
    circles = hough.houghDescent(proc)
    transform_circles(circles, scale, offset)
    draw_circles(circles, cimg)
    cv2.imshow('output',cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#!/bin/python3

import sys
import math
import statistics
import numpy as np
import cv2

from hole import Hole
from hough import Hough

p0 = [-1, -1]
p1 = [-1, -1]
def scale_callback(event, x, y, flags, param):
    global p0, p1
    if event == cv2.EVENT_LBUTTONDOWN:
        p0[0] = x
        p0[1] = y
    elif event == cv2.EVENT_LBUTTONUP:
        p1[0] = x
        p1[1] = y

def get_selection(image, p0, p1, scale):
    output = image[p0[1]:p1[1], p0[0]:p1[0]]

    roi_size = [p1[0] - p0[0], p1[1] - p0[1]]
    new_size = [roi_size[0] * scale, roi_size[1] * scale]

    # scale to a fixed size
    if np.prod(roi_size) > np.prod(new_size):
        output = cv2.resize(output, (int(new_size[0]), int(new_size[1])), interpolation = cv2.INTER_AREA);
    else:
        output = cv2.resize(output, (int(new_size[0]), int(new_size[1])), interpolation = cv2.INTER_CUBIC);

    return output

def transform_circles(circles, scale, offset):
    for c in circles:
        c.x = int(round(float(c.x) / scale + offset[0]))
        c.y = int(round(float(c.y) / scale + offset[1]))
        c.r = int(round(float(c.r) / scale))

def filter_preprocess(image):
    output = image
    output = cv2.blur(output, (7, 7))
    output = cv2.GaussianBlur(output, (7, 7), 1, 1)
    output = cv2.medianBlur(output, 7)
    output = cv2.bilateralFilter(output, 7, 1, 3)
    return output

def preprocess(image):
    output = filter_preprocess(image)
    return output

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
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please provide an image file")
        exit(-1)

    # Create window with freedom of dimensions
    cv2.namedWindow("original"  , cv2.WINDOW_NORMAL)
    cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges"     , cv2.WINDOW_NORMAL)
    cv2.namedWindow("output"    , cv2.WINDOW_NORMAL)

    img = cv2.imread(image_path, 0)
    cv2.imshow("original", img)
    cv2.waitKey(100)

    cv2.setMouseCallback("original", scale_callback)

    # 1: Get the bullet diameter from user
    D = float(input("Input bullet diameter (in):: "))
    R = D/2

    # 2: Get the pixel distance:real distance factor
    global p0, p1
    while (p1[0] == -1 and p1[1] == -1):
        print("Set a pixel to distance conversion factor by clicking on the original image, then hit any key")
        cv2.waitKey(0)
    line_dis = float(math.sqrt(math.pow(p1[0] - p0[0], 2) + math.pow(p1[1] - p0[1], 2)))
    print("Distance from (%d, %d) to (%d, %d) is %d" %(p0[0], p0[1], p1[0], p1[1], int(round(line_dis))))

    # 3: Get the scaling factor for an ROI
    # N: target size
    # C: conversion factor: px/in

    real_dis = float(input("Input actual distance (in):: "))
    C =  line_dis / real_dis
    N = 20
    S = N / (R * C)
    print("Scaling factor: %f" % S)

    # 4: Get the ROI
    p0 = [-1, -1]
    p1 = [-1, -1]
    while (p1[0] == -1 and p1[1] == -1):
        print("Select an ROI, then hit any key")
        cv2.waitKey(0)
    print("Selected (%d, %d) to (%d, %d)" %(p0[0], p0[1], p1[0], p1[1]))

    sel = get_selection(img, p0, p1, S)
    off = p0
    cv2.imshow("roi", sel)

    # 5: Perform preprocessing
    pproc = preprocess(sel)
    cv2.imshow("preprocess", pproc)

    # 6: Do the Hough
    hough = Hough()
    hough.dp        = 1.25
    hough.canny     = 70
    hough.minDist   = int(round(N * 1.00))
    hough.minRadius = int(round(N * 0.75))
    hough.maxRadius = int(round(N * 1.25))
    #hough.accum     = int(round(2 * math.pi * N * 0.75))
    #hough.accum     = int(round(2 * math.pi * N * 0.25))
    hough.accum     = int(round(N))
    print("Hough accumulator threshold: %d" % hough.accum)

    edges   = hough.runCanny(pproc)
    cv2.imshow("edges", edges)

    circles = hough.runHough(pproc)

    # 7: Draw the results
    transform_circles(circles, S, off)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_circles(circles, cimg)
    cv2.imshow("output", cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
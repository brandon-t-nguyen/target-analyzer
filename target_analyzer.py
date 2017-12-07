#!/bin/python3

import sys
import math
import statistics
import copy
import numpy as np
import cv2

from hole import Hole
from hough import Hough
from analyzer import *
from cvhelper import *

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

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result_name = ""
        if len(sys.argv) > 2:
            result_name = "_" + sys.argv[2]
    else:
        print("Please provide an image file")
        exit(-1)

    # Create window with freedom of dimensions
    cv2.namedWindow("original"  , cv2.WINDOW_NORMAL)
    cv2.namedWindow("roi"       , cv2.WINDOW_NORMAL)
    cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges"     , cv2.WINDOW_NORMAL)
    cv2.namedWindow("result"    , cv2.WINDOW_NORMAL)
    cv2.namedWindow("output"    , cv2.WINDOW_NORMAL)

    img = cv2.imread(image_path, 0)
    cv2.imshow("original", img)
    cv2.waitKey(100)

    cv2.setMouseCallback("original", scale_callback)

    params = AnalyzerParams()
    params.dp            = 1.315000
    params.canny         = 21.000000
    params.minDistScale  = 1.990000
    params.minRadScale   = 0.710000
    params.maxRadScale   = 1.245000
    params.accumScale    = 0.179155
    params.gauss_size    = 9
    params.gauss_sigma   = 1.900000
    params.blur_size     = 5
    params.bilat_size    = 9
    params.bilat_sigma1  = 2.400000
    params.bilat_sigma2  = 5.000000
    params.median_size   = 7

    analyzer = Analyzer()
    analyzer.set_image(img)
    analyzer.params = params

    # 1: Get the bullet diameter from user
    D = float(input("Input bullet diameter (in):: "))

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
    analyzer.set_scale(20, D, real_dis, p0, p1)
    print("Scaling factor: %f" % analyzer.S)

    # 4: Get the ROI
    p0 = [-1, -1]
    p1 = [-1, -1]
    while (p1[0] == -1 and p1[1] == -1):
        print("Select an ROI, then hit any key")
        cv2.waitKey(0)
    print("Selected (%d, %d) to (%d, %d)" %(p0[0], p0[1], p1[0], p1[1]))
    analyzer.set_roi(p0, p1)

    # 5: Run
    analyzer.run()
    analyzer.draw()

    cv2.imshow("roi"        , analyzer.sel)
    cv2.imshow("preprocess" , analyzer.pproc)
    cv2.imshow("edges"      , analyzer.edges)
    cv2.imshow("result"     , analyzer.result)
    cv2.imshow("output"     , analyzer.output)

    cv2.waitKey(0)

    # 8: Save results
    cv2.imwrite("out_1%s.png" % result_name, analyzer.sel);
    cv2.imwrite("out_2%s.png" % result_name, analyzer.pproc);
    cv2.imwrite("out_3%s.png" % result_name, analyzer.edges);
    cv2.imwrite("out_4%s.png" % result_name, analyzer.result);

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

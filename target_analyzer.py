#!/bin/python3

import sys
import math
import statistics
import copy
import numpy as np
import cv2

from hole import Hole
from hough import Hough
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

class Analyzer:
    def __init__(self):
        self.R = -1
        self.N = -1
        self.S = -1
        self.C = -1
        self.roi_p1 = [-1, -1]
        self.roi_p2 = [-1, -1]

        self.scale_p1 = [-1, -1]
        self.scale_p2 = [-1, -1]

        # images
        self.img   = None
        self.sel   = None
        self.pproc = None
        self.edges = None
        self.result= None
        self.output= None

        self.holes_roi = None
        self.holes     = None

    def set_image(self, img):
        self.img = img

    def set_roi(self, roi_p1, roi_p2):
        self.roi_p1 = copy.copy(roi_p1)
        self.roi_p2 = copy.copy(roi_p2)

    def set_scale(self, norm_d, diameter, real_dis, scale_p1, scale_p2):
        self.N = norm_d
        self.R = diameter / 2;

        self.scale_p1 = scale_p1[:];
        self.scale_p2 = scale_p2[:];
        line_dis = float(math.sqrt(math.pow(self.scale_p2[0] - self.scale_p1[0], 2) +
                                   math.pow(self.scale_p2[1] - self.scale_p1[1], 2)))
        self.C = line_dis / real_dis;
        self.S = self.N / (self.R * self.C)


    def run(self):
        self.sel = get_selection(self.img, self.roi_p1, self.roi_p2, self.S)

        # 5: Perform preprocessing
        self.pproc = preprocess(self.sel)

        # 6: Do the Hough
        hough = Hough()
        hough.dp        = 1.25
        hough.canny     = 70
        hough.minDist   = int(round(self.N * 2.0))
        hough.minRadius = int(round(self.N * 0.75))
        hough.maxRadius = int(round(self.N * 1.25))
        #hough.accum     = int(round(2 * math.pi * self.N * 0.75))
        #hough.accum     = int(round(2 * math.pi * self.N * 0.25))
        hough.accum     = int(round(self.N * 1.0))

        self.edges     = hough.runCanny(self.pproc)
        self.holes_roi = hough.runHough(self.pproc)
        self.holes = transform_circles(self.holes_roi, self.S, self.roi_p1)


    def draw(self):
        self.result = cv2.cvtColor(self.sel, cv2.COLOR_GRAY2BGR)
        draw_circles(self.holes_roi, self.result, 20)

        self.output = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        draw_circles(self.holes, self.output, 40)

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

    analyzer = Analyzer()
    analyzer.set_image(img)

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

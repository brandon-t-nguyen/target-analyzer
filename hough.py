#!/bin/python3

import math
import statistics
import numpy as np
import cv2

from hole import Hole

class Hough:
    def __init__(self, dp=1.2, minDist=40, canny=100, accum=100, minRadius=20, maxRadius=80):
        self.dp        = dp
        self.minDist   = minDist
        self.canny     = canny
        self.accum     = accum
        self.minRadius = minRadius
        self.maxRadius = maxRadius
    def runHough(self, image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,
                                   self.dp,
                                   self.minDist,
                                   param1=self.canny,
                                   param2=self.accum,
                                   minRadius=self.minRadius,
                                   maxRadius=self.maxRadius)
        output = []
        # check if empty
        if circles is None:
            return output
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            hole = Hole(i[0], i[1], i[2])
            output.append(hole)
        return output
    def runCanny(self, image):
        # use the canny call in hough.cpp
        # cvCanny( img, edges, MAX(canny_threshold/2,1), canny_threshold, 3 );
        edges = cv2.Canny(image, max(self.canny/2, 1), self.canny, 3)
        return edges

    def houghDescent(self, image):
        hough = self
        #hough.minRadius = 5
        #hough.maxRadius = 100

        for i in range(0,5):
            circles = hough.runHough(image)
            if len(circles) == 0:
                break
            statRad = statistics.median(c.r for c in circles)
            print ("Min radius: %d, Max radius: %d" %(hough.minRadius, hough.maxRadius))
            hough.minRadius = int(round(hough.minRadius + abs(statRad - hough.minRadius) / 8))
            hough.maxRadius = int(round(hough.maxRadius - abs(statRad - hough.maxRadius) / 8))

        return hough.runHough(image)
import sys
import math
import statistics
import copy
import numpy as np
import cv2

from hole import Hole
from hough import Hough
from cvhelper import *

class AnalyzerParams:
    def __init__(self):
        self.dp             = 1.25
        self.canny          = 70
        self.minDistScale   = 2.0
        self.minRadScale    = 0.75
        self.maxRadScale    = 1.25
        self.accumScale     = 1 / (2 * math.pi)

        # preprocessing
        self.gauss_size     = 3
        self.gauss_sigma    = 1
        self.blur_size      = 5
        self.bilat_size     = 9
        self.bilat_sigma1   = 3
        self.bilat_sigma2   = 1
        self.median_size    = 7

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

        self.params = AnalyzerParams()

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
        self.pproc = preprocess(self.sel, self.params)

        # 6: Do the Hough
        hough = Hough()
        hough.dp        = self.params.dp
        hough.canny     = self.params.canny
        hough.minDist   = int(round(self.N * self.params.minDistScale))
        hough.minRadius = int(round(self.N * self.params.minRadScale))
        hough.maxRadius = int(round(self.N * self.params.maxRadScale))
        hough.accum     = int(round(self.N * 2 * math.pi * self.params.accumScale))

        self.edges     = hough.runCanny(self.pproc)
        self.holes_roi = hough.runHough(self.pproc)
        self.holes = transform_circles(self.holes_roi, self.S, self.roi_p1)


    def draw(self):
        hole_size = max(h.r for h in self.holes_roi)
        self.result = cv2.cvtColor(self.sel, cv2.COLOR_GRAY2BGR)
        draw_circles(self.holes_roi, self.result, hole_size)

        hole_size = max(h.r for h in self.holes)
        self.output = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        draw_circles(self.holes, self.output, hole_size)

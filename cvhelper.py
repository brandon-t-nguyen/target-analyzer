import sys
import math
import statistics
import copy
import numpy as np
import cv2
from hole import Hole
from analyzer import *

def get_selection(image, p0, p1, scale):
    output = image[p0[1]:p1[1], p0[0]:p1[0]]

    roi_size = [p1[0] - p0[0], p1[1] - p0[1]]
    new_size = [int(round(roi_size[0] * scale)), int(round(roi_size[1] * scale))]

    # scale to a fixed size
    if np.prod(roi_size) > np.prod(new_size):
        output = cv2.resize(output, (new_size[0], new_size[1]), interpolation = cv2.INTER_AREA);
    else:
        output = cv2.resize(output, (new_size[0], new_size[1]), interpolation = cv2.INTER_CUBIC);

    return output

def transform_circles(circles, scale, offset):
    holes = copy.deepcopy(circles)
    for h in holes:
        h.x = int(round(float(h.x) / scale + offset[0]))
        h.y = int(round(float(h.y) / scale + offset[1]))
        h.r = int(round(float(h.r) / scale))

    return holes

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

def morph_preprocess(image, p):
    output = image
    output = cv2.GaussianBlur(output, (3, 3), 1, 1)
    output = cv2.bilateralFilter(output, 3, 1, 1)

    # 11
    output = cv2.adaptiveThreshold(output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

    n = 3
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    output = m_close(output, k_rect, n)

    return output

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

def filter_preprocess(image, p):
    output = image
    output = cv2.GaussianBlur(output, (p.gauss_size, p.gauss_size), p.gauss_sigma, p.gauss_sigma)
    output = cv2.blur(output, (p.blur_size, p.blur_size))
    output = cv2.bilateralFilter(output, p.bilat_size, p.bilat_sigma1, p.bilat_sigma2)
    output = cv2.medianBlur(output, p.median_size)

    # somehow this helps a lot
    output = sharpen(output)
    output = cv2.bilateralFilter(output, p.bilat_size, p.bilat_sigma1, p.bilat_sigma2)
    output = cv2.bilateralFilter(output, p.bilat_size, p.bilat_sigma1, p.bilat_sigma2)
    output = cv2.bilateralFilter(output, p.bilat_size, p.bilat_sigma1, p.bilat_sigma2)
    return output

def preprocess(image, p):
    output = filter_preprocess(image, p)
    return output

def draw_circle(c, cimg):
    # draw the outer circle
    cv2.circle(cimg,(c.x,c.y),c.r+5,(0,0,0),-2)
    cv2.circle(cimg,(c.x,c.y),c.r,(0,255,0),-2)

    # draw the center of the circle
    cv2.circle(cimg,(c.x,c.y),2,(0,0,255),3)

def draw_cross(x, y, thicc, size, cimg):
        # back
        p1 = (x - thicc * 2, y - size - thicc * 2)
        p2 = (x + thicc * 2, y + size + thicc * 2)
        cv2.rectangle(cimg, p1, p2, (0,0,0), -1)
        p1 = (x - size - thicc * 2, y - thicc * 2)
        p2 = (x + size + thicc * 2, y + thicc * 2)
        cv2.rectangle(cimg, p1, p2, (0,0,0), -1)

        # fore
        p1 = (x - thicc * 1, y - size - thicc * 1)
        p2 = (x + thicc * 1, y + size + thicc * 1)
        cv2.rectangle(cimg, p1, p2, (0,0,255), -1)
        p1 = (x - size - thicc * 1, y - thicc * 1)
        p2 = (x + size + thicc * 1, y + thicc * 1)
        cv2.rectangle(cimg, p1, p2, (0,0,255), -1)

def draw_circles(circles, cimg, cross_size):
    n = len(circles)
    i = 0
    for c in circles:
        draw_circle(c, cimg)
        #print('%d: (%d, %d)' %(i, c.x, c.y))
        i += 1

    # draw the mean as a cross
    if n > 0:
        meanX = np.uint16(statistics.mean(c.x for c in circles))
        meanY = np.uint16(statistics.mean(c.y for c in circles))

        draw_cross(meanX, meanY, int(round(cross_size/8)), cross_size, cimg)

        #print('Mean: (%d, %d)' %(meanX, meanY))

import sys
import math
import statistics
import copy
import numpy as np
import cv2
from hole import Hole

def get_selection(image, p0, p1, scale):
    output = image[p0[1]:p1[1], p0[0]:p1[0]]

    roi_size = [p1[0] - p0[0], p1[1] - p0[1]]
    new_size = [int(round(roi_size[0] * scale)), int(round(roi_size[1] * scale))]

    print("ROI size:")
    print(roi_size)
    print("New size:")
    print(new_size)

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
        print('%d: (%d, %d)' %(i, c.x, c.y))
        i += 1

    # draw the mean as a cross
    if n > 0:
        meanX = np.uint16(statistics.mean(c.x for c in circles))
        meanY = np.uint16(statistics.mean(c.y for c in circles))

        draw_cross(meanX, meanY, int(round(cross_size/8)), cross_size, cimg)

        print('Mean: (%d, %d)' %(meanX, meanY))

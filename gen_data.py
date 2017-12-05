#!/bin/python3

import sys
import math
import statistics
import numpy as np
import cv2

from hole import Hole
from hough import Hough
from cvhelper import draw_circle

# how this works:
# gen_data.py <image file path> <database csv file path>

# states
# 0: select sel_p1
# 1: select sel_p2
# 2: select roi_p1
# 3: select roi_p2
# 4: select points
img = None
state = 0
sel_p1 = [-1, -1]
sel_p2 = [-1, -1]
roi_p1 = [-1, -1]
roi_p2 = [-1, -1]
points = []
real_dis = -1
def click_callback(event, x, y, flags, param):
    global img, state, sel_p1, sel_p2, roi_p1, roi_p2, points, real_dis
    if event == cv2.EVENT_LBUTTONDOWN:
        if state == 0:
            sel_p1 = [x, y]
            state += 1
            print("Click scale setting point 2")
        elif state == 1:
            sel_p2 = [x, y]
            state += 1
            real_dis = float(input("Input the real distance (in):: "))
            print("Click roi setting point 1")
        elif state == 2:
            roi_p1 = [x, y]
            state += 1
            print("Click roi setting point 2")
        elif state == 3:
            roi_p2 = [x, y]
            state += 1
            print("Click the holes then hit any key")
        elif state == 4:
            p = [x, y]
            points.append(p)
            # draw marker
            circle = Hole(x, y, 20)
            draw_circle(circle, img)
            cv2.imshow("original", img)

def main():
    global img, state, sel_p1, sel_p2, roi_p1, roi_p2, points, real_dis
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        data_path  = sys.argv[2]
    else:
        print("Please provide an image file")
        exit(-1)

    # Create window with freedom of dimensions
    cv2.namedWindow("original"  , cv2.WINDOW_NORMAL)

    img = cv2.imread(image_path)
    cv2.imshow("original", img)
    cv2.waitKey(100)
    cv2.setMouseCallback("original", click_callback)

    print("Click scale setting point 1")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    diameter = float(input("Input projectile diameter (in):: "))
    target_type = input("Input target type (reactive, nra, sighting):: ")

    # open the database file
    with open(data_path, "a") as f:
        output_string = ""
        output_string += image_path + ','
        output_string += str(diameter) + ','
        output_string += target_type + ','
        output_string += str(real_dis) + ','

        sel_string = "\"%d %d,%d %d\"" %(sel_p1[0], sel_p1[1], sel_p2[0], sel_p2[1])
        output_string += sel_string + ','

        roi_string = "\"%d %d,%d %d\"" %(roi_p1[0], roi_p1[1], roi_p2[0], roi_p2[1])
        output_string += roi_string + ','

        if len(points) > 0:
            points_string = ""
            for i in range(0, len(points)):
                point = points[i]
                point_string = "%d %d," %(point[0], point[1])
                points_string += point_string
            # strip off the last comma
            output_string += '"' + points_string[:-1] + '"'
        else:
            print("No points selected")

        # strip off the last comma
        if output_string[-1] == ',':
            output_string = output_string[:-1]
        f.write(output_string + "\n")

if __name__ == "__main__":
    main()

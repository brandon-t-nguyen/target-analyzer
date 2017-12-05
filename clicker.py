#!/bin/python3

import sys
import math
import statistics
import numpy as np
import cv2

from hole import Hole
from hough import Hough

def scale_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("Clicked at [%d, %d]" %(x, y))
        print("%d %d," %(x, y), end='')

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please provide an image file")
        exit(-1)

    # Create window with freedom of dimensions
    cv2.namedWindow("original"  , cv2.WINDOW_NORMAL)

    img = cv2.imread(image_path, 0)
    cv2.imshow("original", img)
    cv2.waitKey(100)
    cv2.setMouseCallback("original", scale_callback)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#!/bin/python3

import sys
import csv
import math
import statistics
import copy
import numpy as np
import cv2

from target_analyzer import Analyzer
from hole import Hole

class Stats:
    def __init__(self):
        self.pos = []
        self.neg = []
        # successfully found
        self.true_pos = []
        # said there was a hole where there isn't
        self.false_pos = []
        # said there wasn't a hole where there is
        self.false_neg = []

    def num_pos(self):
        return len(self.pos)

    def num_true_pos(self):
        return len(self.true_pos)

    def num_false_pos(self):
        return len(self.false_pos)

    def num_false_neg(self):
        return len(self.false_neg)

    def get_tpr(self):
        return self.num_true_pos() / self.num_pos()

    def get_fnr(self):
        return self.num_false_neg() / self.num_pos()

def check_holes(check, orig_holes):
    holes = copy.deepcopy(orig_holes)

    stats = Stats()

    stats.pos = copy.copy(check)

    for c in check:
        hole_found = False
        for h in holes:
            if c.is_inside(h):
                hole_found = True
                holes[:] = [x for x in holes if x != h]
                break
        if not hole_found:
            #print("Hole [%d, %d] is not present" %(c.x, c.y))
            stats.false_neg.append(c)
        else:
            stats.true_pos.append(c)


    for h in holes:
        #print("False hole: [%d, %d]" %(h.x, h.y))
        stats.false_pos.append(h)

    return stats

def get_coord(string_pair):
    coords_strings = string_pair.split(" ")
    pair = [int(coords_strings[0]), int(coords_strings[1])];
    return pair

def main():
    dataset_path = "./dataset.csv"
    show_mode = False
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == "--show":
                show_mode = True

    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)

        # run for each test
        n = 1
        for row in reader:
            path     = row["file"]
            diameter = float(row["diameter"])
            real_dis = float(row["scale"])

            scale_p = row["scale_p"]
            scale_pairs = scale_p.split(",")
            scale_p1 = get_coord(scale_pairs[0])
            scale_p2 = get_coord(scale_pairs[1])

            roi_p = row["roi_p"]
            roi_pairs = roi_p.split(",")
            roi_p1 = get_coord(roi_pairs[0])
            roi_p2 = get_coord(roi_pairs[1])

            holes_string = row["holes"]
            holes_pairs  = holes_string.split(",")
            check = []
            for pair in holes_pairs:
                #coord_string = pair.split(" ")
                #coord = Hole(int(coord_string[0]), int(coord_string[1]), 0)
                coord = get_coord(pair)
                coord_hole = Hole(coord[0], coord[1], 0)
                check.append(coord_hole);

            analyzer = Analyzer()
            img = cv2.imread(path, 0)
            analyzer.set_image(img)
            analyzer.set_scale(20, diameter, real_dis, scale_p1, scale_p2)
            analyzer.set_roi(roi_p1, roi_p2)
            analyzer.run()

            stats = check_holes(check, analyzer.holes)
            print("Testcase #%d" %(n))
            print("True positive rate: %0.3f, %d/%d" %(stats.get_tpr(), stats.num_true_pos(), stats.num_pos()))
            print("False negative rate: %0.3f, %d/%d" %(stats.get_fnr(), stats.num_false_neg(), stats.num_pos()))
            print("False positives: %d" %(stats.num_false_pos()))
            print("")
            n += 1

            if show_mode:
                analyzer.draw()
                cv2.namedWindow("original"  , cv2.WINDOW_NORMAL)
                cv2.namedWindow("roi"       , cv2.WINDOW_NORMAL)
                cv2.namedWindow("preprocess", cv2.WINDOW_NORMAL)
                cv2.namedWindow("edges"     , cv2.WINDOW_NORMAL)
                cv2.namedWindow("result"    , cv2.WINDOW_NORMAL)
                cv2.namedWindow("output"    , cv2.WINDOW_NORMAL)
                cv2.imshow("original"   , analyzer.img)
                cv2.imshow("roi"        , analyzer.sel)
                cv2.imshow("preprocess" , analyzer.pproc)
                cv2.imshow("edges"      , analyzer.edges)
                cv2.imshow("result"     , analyzer.result)
                cv2.imshow("output"     , analyzer.output)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        f.close()

if __name__ == "__main__":
    main()

#!/bin/python3

import sys
import csv
import math
import statistics
import copy
import random
import time
from operator import attrgetter
import numpy as np
import cv2

from analyzer import *
from hole import Hole

class OverallStats:
    def __init__(self, target_type = ""):
        self.target_type    = target_type
        self.total_pos  = 0
        self.total_true_pos  = 0
        self.total_false_neg = 0
        self.total_false_pos = 0

    def get_tpr(self):
        if self.total_pos == 0:
            return -1
        return self.total_true_pos / self.total_pos
    def get_fnr(self):
        if self.total_pos == 0:
            return -1
        return self.total_false_neg / self.total_pos
    def get_tdr(self):
        if self.total_true_pos + self.total_false_pos == 0:
            return 0
        return self.total_true_pos / (self.total_true_pos + self.total_false_pos)
    def get_fdr(self):
        if self.total_true_pos + self.total_false_pos == 0:
            return 0
        return self.total_false_pos / (self.total_true_pos + self.total_false_pos)

    def get_precision(self):
        return self.get_tdr()
    def get_recall(self):
        if self.total_true_pos + self.total_false_neg == 0:
            return -1
        return self.total_true_pos / (self.total_true_pos + self.total_false_neg)
    def get_f1(self):
        if self.get_precision() == 0 or self.get_recall() == 0:
            return -1
        return 2 / ((1/self.get_precision()) + (1/self.get_recall()))
    def get_fb(self, b):
        if self.get_precision() == 0 or self.get_recall() == 0:
            return -1
        b2 = b * b
        pre = self.get_precision()
        rec = self.get_recall()
        return (1 + b2) * pre * rec / (b2 * pre + rec)

class Stats:
    def __init__(self):
        self.id          = None
        self.target_type = None
        self.diameter    = None

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

    def get_fdr(self):
        if self.num_true_pos() + self.num_false_pos() == 0:
            return 0
        return self.num_false_pos() / (self.num_true_pos() + self.num_false_pos())

    def get_tdr(self):
        if self.num_true_pos() + self.num_false_pos() == 0:
            return 0
        return self.num_true_pos() / (self.num_true_pos() + self.num_false_pos())

    def get_precision(self):
        return get_tdr(self)
    def get_recall(self):
        return self.num_true_pos() / (self.num_true_pos() + self.num_false_neg())

def check_holes(check, orig_holes):
    holes = copy.deepcopy(orig_holes)
    for h in holes:
        # make the hole slightly bigger for leniency
        h.r *= 1.25

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

def run_test(row, params):
    path     = row["file"]
    target_type = row["type"]
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
        coord = get_coord(pair)
        coord_hole = Hole(coord[0], coord[1], 0)
        check.append(coord_hole);

    analyzer = Analyzer()
    analyzer.params = params
    img = cv2.imread(path, 0)
    analyzer.set_image(img)
    analyzer.set_scale(20, diameter, real_dis, scale_p1, scale_p2)
    analyzer.set_roi(roi_p1, roi_p2)
    analyzer.run()
    stats = check_holes(check, analyzer.holes)

    stats.diameter = diameter
    stats.target_type = target_type
    return stats, analyzer

def print_stats(overalls, overall, beta):
    for o in overalls:
        print(o.target_type + "::")
        print("---------------")
        print("True positive rate (recall)     : %0.3f, %d/%d"
                %(o.get_tpr(), o.total_true_pos, o.total_pos))
        print("Pos predictive value (precision): %0.3f, %d/%d"
                %(o.get_tdr(), o.total_true_pos,
                  o.total_false_pos + o.total_true_pos))
        print("False positives: %d"
                %(o.total_false_pos))
        print("F1 score: %0.3f"
                %(o.get_f1()))
        print("F%0.3f score: %0.3f"
                %(beta, o.get_fb(beta)))
        print("")

    # print out overall stats: evenly weighted
    print("Unweighted average TPR (recall)   : %0.3f" %(statistics.mean(o.get_tpr() for o in overalls)))
    print("Unweighted average PPV (precision): %0.3f" %(statistics.mean(o.get_tdr() for o in overalls)))
    print("Unweighted average F1 : %0.3f" %(statistics.mean(o.get_f1() for o in overalls)))
    print("Unweighted average F%0.3f : %0.3f" %(beta, statistics.mean(o.get_fb(beta) for o in overalls)))
    print("")

    # print out unbiased stats: unweighted
    print("Weighted average TPR (recall)   : %0.3f, %d/%d"
                %(overall.get_tpr(), overall.total_true_pos, overall.total_pos))
    print("Weighted average PPV (precision): %0.3f, %d/%d"
            %(overall.get_tdr(), overall.total_true_pos,
              overall.total_false_pos + overall.total_true_pos))
    print("Weighted average F1 : %0.3f" %(overall.get_f1()))
    print("Weighted average F%0.3f : %0.3f" %(beta, overall.get_fb(beta)))

    print("False positives: %d" %(overall.total_false_pos))

def calculate_overalls(results):
    # calculate the stats
    reactive = OverallStats("reactive")
    nra      = OverallStats("nra")
    sighting = OverallStats("sighting")
    overalls = [reactive, nra, sighting]
    for result in results:
        for overall in overalls:
            if overall.target_type in result.target_type:
                dest = overall
                break
        dest.total_pos += result.num_pos()
        dest.total_true_pos  += result.num_true_pos()
        dest.total_false_neg += result.num_false_neg()
        dest.total_false_pos += result.num_false_pos()
    return overalls

def calculate_overall(overalls):
    overall = OverallStats()
    overall.total_pos = sum(o.total_pos for o in overalls)
    overall.total_true_pos = sum(o.total_true_pos for o in overalls)
    overall.total_false_neg  = sum(o.total_false_neg for o in overalls)
    overall.total_false_pos = sum(o.total_false_pos for o in overalls)
    return overall

class SearchRun:
    def __init__(self, params, overall, overalls):
        self.params  = copy.deepcopy(params)
        self.overall = overall
        self.overalls= overalls

# random mutator
class Mutator:
    def __init__(self):
        self.rng = random.Random()
        self.gauss_size_d   = 2
        self.gauss_sigma_d  = 0.1
        self.blur_size_d    = 2
        self.bilat_size_d   = 2
        self.bilat_sigma1_d = 0.1
        self.bilat_sigma2_d = 0.1
        self.median_size_d  = 2

        self.canny_d    = 1
        self.minDistS_d = 0.005
        self.minRadS_d  = 0.005
        self.maxRadS_d  = 0.005
        self.accumS_d   = 0.005
    def set_seed(self, seed):
        self.rng.seed(seed)
    def direction(self):
        direction = int(math.floor(self.rng.random() * 3)) - 1
        return direction
    def mutate(self, params):
        direction = self.direction()
        if direction == -1 and params.gauss_size <= 1:
            direction = 0
        params.gauss_size += self.gauss_size_d * direction

        direction = self.direction()
        if direction == -1 and params.gauss_sigma <= 0.1:
            direction = 0
        params.gauss_sigma += self.gauss_sigma_d * direction

        direction = self.direction()
        if direction == -1 and params.blur_size <= 1:
            direction = 0
        params.blur_size += self.blur_size_d * direction

        direction = self.direction()
        if direction == -1 and params.blur_size <= 1:
            direction = 0
        params.blur_size += self.blur_size_d * direction

        direction = self.direction()
        if direction == -1 and params.bilat_sigma1 <= 0.1:
            direction = 0
        params.bilat_sigma1 += self.bilat_sigma1_d * direction

        direction = self.direction()
        if direction == -1 and params.bilat_sigma2 <= 0.1:
            direction = 0
        params.bilat_sigma2 += self.bilat_sigma2_d * direction

        direction = self.direction()
        if direction == -1 and params.blur_size <= 1:
            direction = 0
        params.blur_size += self.blur_size_d * direction

        direction = self.direction()
        if direction == -1 and params.canny <= 1:
            direction = 0
        params.canny += self.canny_d * direction

        direction = self.direction()
        if direction == -1 and params.minDistScale <= 1.0:
            direction = 0
        params.minDistScale += self.minDistS_d * direction

        direction = self.direction()
        if direction == -1 and params.minRadScale <= 0.25:
            direction = 0
        params.minRadScale += self.minRadS_d * direction

        direction = self.direction()
        if direction == -1 and params.maxRadScale <= 1.00:
            direction = 0
        params.maxRadScale += self.maxRadS_d * direction

        direction = self.direction()
        if direction == -1 and params.accumScale <= 0.01:
            direction = 0
        params.accumScale += self.accumS_d * direction


# parameter search algorithm
def run_search(dataset_path, beta, num, timed, batch):
    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)
        mutator = Mutator()
        params = AnalyzerParams()
        params.gauss_size     = 3
        params.gauss_sigma    = 1
        params.blur_size      = 5
        params.bilat_size     = 9
        params.bilat_sigma1   = 3
        params.bilat_sigma2   = 1
        params.median_size    = 7

        runs = []
        i = 0
        end = time.time() + num
        while (not timed and i < num) or (timed and time.time() < end):
            # every batch of parameters, find the max and start from there
            i += 1
            if i % batch == 0:
                best = max(runs, key=lambda run:run.overall.get_fb(beta))
                runs = []
                runs.append(best)
                params = copy.deepcopy(best.params)

            # run for each test
            results = []
            n = 1
            for row in reader:
                stats, analyzer = run_test(row, params)
                n += 1
                stats.id    = n
                stats.img   = None
                stats.sel   = None
                stats.pproc = None
                stats.edges = None
                stats.result= None
                stats.output= None
                results.append(stats)
            overalls = calculate_overalls(results)
            overall  = calculate_overall(overalls)
            runs.append(SearchRun(params, overall, overalls))

            # mutate the parameters
            mutator.mutate(params)

            print("iter %d" %i)
            f.seek(0)
            reader = csv.DictReader(f)
    f.close()
    print("")
    max_run = max(runs, key=lambda run:run.overall.get_fb(beta))
    print_stats(max_run.overalls, max_run.overall, beta)
    print("")
    max_run.params.print_out()
    print("")

    #for run in runs:
    #    print_stats(run.overalls, run.overall, beta)
    #    print("")
    #    run.params.print_out()
    #    print("")

def main():
    dataset_path = "./dataset.csv"
    show_mode  = False
    print_each = False
    search    = False
    batch     = 50
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        if "--show" in sys.argv:
            show_mode  = True
        if "--print" in sys.argv:
            print_each = True
        if "--search" in sys.argv:
            search    = True
            timed     = False
            num       = -1
            beta = float(sys.argv[sys.argv.index("--search")+1])
            num_string = sys.argv[sys.argv.index("--search")+2]
            if num_string[-1] == 's':
                num = float(num_string[:-1])
                timed = True
            elif num_string[-1] == 'm':
                num = float(num_string[:-1]) * 60
                timed = True
            elif num_string[-1] == 'h':
                num = float(num_string[:-1]) * 3600
                timed = True
            else:
                num = int(num_string)
        if "--batch" in sys.argv:
            batch = int(sys.argv[sys.argv.index("--batch")+1])

    if search:
        if timed:
            print("Running for %0.3f seconds" % num)
        else:
            print("Running for %d iterations" % num)
        run_search(dataset_path, beta, num, timed, batch)
        exit(0)

    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)

        # run for each test
        params = AnalyzerParams()

        params.dp            = 1.250000
        params.canny         = 72.000000
        params.minDistScale  = 2.015000
        params.minRadScale   = 0.770000
        params.maxRadScale   = 1.235000
        params.accumScale    = 0.149155
        params.gauss_size    = 5
        params.gauss_sigma   = 1.300000
        params.blur_size     = 7
        params.bilat_size    = 9
        params.bilat_sigma1  = 3.400000
        params.bilat_sigma2  = 1.400000
        params.median_size   = 7

        results = []
        n = 1
        for row in reader:
            stats, analyzer = run_test(row, params)
            if print_each:
                print("Testcase #%d" %(n))
                print("True positive rate: %0.3f, %d/%d"
                        %(stats.get_tpr(), stats.num_true_pos(), stats.num_pos()))
                print("False negative rate: %0.3f, %d/%d"
                        %(stats.get_fnr(), stats.num_false_neg(), stats.num_pos()))
                print("True discovery rate: %0.3f, %d/%d"
                        %(stats.get_tdr(), stats.num_true_pos(),
                          stats.num_true_pos() + stats.num_false_pos()))
                print("False discovery rate: %0.3f, %d/%d"
                        %(stats.get_fdr(), stats.num_false_pos(),
                          stats.num_true_pos() + stats.num_false_pos()))
                print("False positives: %d"
                        %(stats.num_false_pos()))
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
            stats.id    = n
            stats.img   = None
            stats.sel   = None
            stats.pproc = None
            stats.edges = None
            stats.result= None
            stats.output= None
            results.append(stats)
        f.close()

        overalls = calculate_overalls(results)
        overall  = calculate_overall(overalls)
        print_stats(overalls, overall, 0.5)

if __name__ == "__main__":
    main()

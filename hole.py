#!/bin/python3

import math

class Hole:
    def __init__(self, x=-1, y=-1, r=-1):
        self.x = x
        self.y = y
        self.r = r

    def distance(self, other):
        return math.sqrt(math.pow(float(self.x) - float(other.x), 2) + math.pow(float(self.y) - float(other.y), 2))

    # check if this hole is in another hole
    def is_inside(self, other):
        dist = self.distance(other)
        if dist + float(self.r) <= float(other.r):
            return True
        else:
            return False

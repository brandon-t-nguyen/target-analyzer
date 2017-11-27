#!/bin/python3

import math

class Hole:
    def __init__(self, x=-1, y=-1, r=-1):
        self.x = x
        self.y = y
        self.r = r

    def distance(self, other):
        return math.sqrt(math.pow(float(self.x) - float(other.x), 2) + math.pow(float(self.y) - float(other.y), 2))

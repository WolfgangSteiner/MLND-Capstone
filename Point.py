import random

class Point(object):
    """ Simple class to represent a point, size or distance in 2d space. """
    def __init__(self, x,y):
        self.x = x
        self.y = y


    def __add__(self, p):
        return Point(self.x+p.x, self.y+p.y)


    def __sub__(self, p):
        return Point(self.x-p.x, self.y-p.y)


    def __mul__(self, factor):
        return Point(self.x * factor, self.y * factor)

    __rmul__ = __mul__


    def unscale(self, scale):
        return Point(self.x / scale.x, self.y / scale.y)


    def scale(self, scale):
        return Point(self.x * scale.x, self.y * scale.y)


    @staticmethod
    def random(p):
        return Point(random.random() * p.x, random.random() * p.y)

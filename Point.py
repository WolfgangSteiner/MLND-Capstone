class Point(object):
    """ Simple class to represent a point, size or distance in 2d space. """
    def __init__(self, x,y):
        self.x = x
        self.y = y


    def __add__(self, p):
        return Point(self.x+p.x, self.y+p.y)


    def unscale(self, scale):
        return Point(self.x / scale.x, self.y / scale.y)

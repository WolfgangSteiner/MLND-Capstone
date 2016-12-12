from Point import Point

class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)


    def p1(self):
        return Point(self.x1, self.y1)


    def p2(self):
        return Point(self.x2, self.y2)


    def unscale(self, scale):
        return Rectangle(self.x1 / scale.x, self.y1 / scale.y, self.x2 / scale.x, self.y2 / scale.y)


    def as_array(self):
        return [self.x1, self.y1, self.x2, self.y2]


    @staticmethod
    def from_points(p1, p2):
        return Rectangle(p1.x, p1.y, p2.x, p2.y)


    @staticmethod
    def from_point_and_size(pos, size):
        return Rectangle.from_points(pos, pos + size)


    @staticmethod
    def from_center_and_size(center, size):
        return Rectangle.from_points(center - 0.5 * size, center + 0.5 * size)


    def intersects_horizontally(self, other_rect):
        return not self.x2 < other_rect.x1 and not self.x1 > other_rect.x2


    def intersects_vertically(self, other_rect):
        return not self.y2 < other_rect.y1 and not self.y1 > other_rect.y2


    def intersects(self, other_rect):
        return self.intersects_horizontally(other_rect) and self.intersects_vertically(other_rect)


    def union_with(self, other_rect):
        self.x1 = min(self.x1, other_rect.x1)
        self.x2 = max(self.x2, other_rect.x2)
        self.y1 = min(self.y1, other_rect.y1)
        self.y2 = max(self.y2, other_rect.y2)


    def intersect_with(self, other_rect):
        self.x1 = min(self.x1, other_rect.x1)
        self.x2 = max(self.x2, other_rect.x2)
        self.y1 = min(self.y1, other_rect.y1)
        self.y2 = max(self.y2, other_rect.y2)


    def union(self, other_rect):
        r = Rectangle(self.x1,self.y1,self.x2,self.y2)
        r.union_with(other_rect)
        return r


    def center(self):
        return 0.5 * (self.p1() + self.p2())


    def __repr__(self):
        return "(%.2f, %.2f, %.2f, %.2f)" % (self.x1,self.y1,self.x2,self.y2)

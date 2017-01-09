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


    def scale(self, scale):
        return Rectangle(self.x1 * scale.x, self.y1 * scale.y, self.x2 * scale.x, self.y2 * scale.y)


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


    def calc_overlap(self, other_rect):
        if not self.intersects(other_rect) or self.area() == 0.0:
            return 0.0
        else:
            return self.intersect(other_rect).area() / self.area()


    def contains(self, r):
        return self.x1 <= r.x1 and self.x2 >= r.x2 and self.y1 <= r.y1 and self.y2 >= r.y2


    def contains_vertically(self, r):
        return self.y1 <= r.y1 and self.y2 >= r.y2


    def shrink(self, m):
        return Rectangle(self.x1 + m, self.y1 + m, self.x2 - m, self.y2 - m)


    def union_with(self, other_rect):
        self.x1 = min(self.x1, other_rect.x1)
        self.x2 = max(self.x2, other_rect.x2)
        self.y1 = min(self.y1, other_rect.y1)
        self.y2 = max(self.y2, other_rect.y2)


    def intersect_with(self, other_rect):
        if self.intersects(other_rect):
            self.x1 = max(self.x1, other_rect.x1)
            self.x2 = min(self.x2, other_rect.x2)
            self.y1 = max(self.y1, other_rect.y1)
            self.y2 = min(self.y2, other_rect.y2)
        else:
            self.x1 = self.x2 = self.y1 = self.y2 = 0


    def union(self, other_rect):
        r = Rectangle(self.x1,self.y1,self.x2,self.y2)
        r.union_with(other_rect)
        return r


    def intersect(self, other_rect):
        r = Rectangle(self.x1,self.y1,self.x2,self.y2)
        r.intersect_with(other_rect)
        return r


    def size(self):
        return self.p2() - self.p1()


    def width(self):
        return self.size().x


    def height(self):
        return self.size().y


    def center(self):
        return 0.5 * (self.p1() + self.p2())


    def area(self):
        s = self.size()
        return s.x * s.y


    def __imul__(self, a):
        self.x1 *= a
        self.x2 *= a
        self.y1 *= a
        self.y2 *= a
        return self


    def __repr__(self):
        return "(%.2f, %.2f, %.2f, %.2f)" % (self.x1,self.y1,self.x2,self.y2)

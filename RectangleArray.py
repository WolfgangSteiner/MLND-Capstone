from Rectangle import Rectangle

class RectangleArray(object):
    """ An array of Rectangles used to compute the resulting bounding boxes. """
    def __init__(self, other_array = [], overlap=0.0):
        self.list = list(other_array)
        self.separate_list = list(other_array)
        self.overlap = overlap


    def add(self, rect):
        """Add a rectangle to the array.
        Args:
            rect: the rect to add to the array
        Returns:
            True if the rect was joined with an existing rect in the array"""

        self.separate_list.append(rect)

        for r in self.list:
            if r.calc_overlap(rect) > self.overlap:
                r.union_with(rect)
                return True

        self.list.append(rect)
        return False


    def finalize(self):
        """Finalize the rectangle calculation."""

        result = RectangleArray(self.list, self.overlap)
        current_list = list(self.list)

        while True:
            did_join = False
            result = RectangleArray(overlap=self.overlap)
            for r in current_list:
                did_join |= result.add(r)

            current_list = list(result.list)

            if not did_join:
                break

        self.list = result.list


    def __iter__(self):
        return self.list.__iter__()


if __name__ == "__main__":
    a = RectangleArray()
    a.add(Rectangle(0,0,1,1))
    print a.list
    a.add(Rectangle(4,0,6,1))
    print a.list
    a.add(Rectangle(0.5,0,3,1))
    print a.list
    a.add(Rectangle(2.5,0,4.5,1))
    print a.list
    a.add(Rectangle(8,0,10,1))
    a.finalize()
    print a.list

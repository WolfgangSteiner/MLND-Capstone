from Rectangle import Rectangle

class RectangleArray(object):
    """ An array of Rectangles used to compute the resulting bounding boxes. """
    def __init__(self, other_array = []):
        self.list = list(other_array)


    def add(self, rect):
        """Add a rectangle to the array.
        Args:
            rect: the rect to add to the array
        Returns:
            True if the rect was joined with an existing rect in the array"""

        for r in self.list:
            if r.intersects(rect):
                r.union_with(rect)
                return True

        self.list.append(rect)
        return False


    def finalize(self):
        """Finalize the rectangle calculation."""

        result = RectangleArray(self.list)
        current_list = list(self.list)

        while True:
            did_join = False
            result = RectangleArray()
            for r in current_list:
                did_join |= result.add(r)

            current_list = list(result.list)

            if not did_join:
                break

        self.list = result.list


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
    a.finalize()
    print a.list

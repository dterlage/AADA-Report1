EPS = 1e-12

class Point:
    """Represents a point in the plane.
    Contains a dimension field and a list of coordinates for each dimension
    """
    def __init__(self, dimension, coords=None):
        self.dimension = dimension
        if coords is None:
            self.coords = [0. for _ in range(dimension)]
        else:
            assert len(coords) == dimension
            self.coords = coords

    def __str__(self):
        return " ".join("{}".format(c) for c in self.coords)

    def __eq__(self, other):
        return self.sq_distance_to(other) < EPS

    def __ne__(self, other):
        return not self.__eq__(other)

    def sq_distance_to(self, other):
        assert self.dim_match(other)

        return sum((self.coords[i] - other.coords[i]) ** 2 for i in range(self.dimension))

    def add(self, other):
        assert self.dim_match(other)

        for i in range(self.dimension):
            self.coords[i] += other.coords[i]

    def sub(self, other):
        assert self.dim_match(other)

        for i in range(self.dimension):
            self.coords[i] -= other.coords[i]

    def mul(self, x):
        self.coords = [coord * x for coord in self.coords]

    def div(self, x):
        self.coords = [coord / x for coord in self.coords]

    def dim_match(self, other):
        return self.dimension == other.dimension

    def get(self, i):
        return self.coords[i]


class ClusterPoint(Point):
    """Point containing an additional clustering label"""
    def __init__(self, dimension, coords=None, label=-1):
        super().__init__(dimension, coords)

        self.cluster_label = label

    def __str__(self):
        return "{} {}".format(super().__str__(), self.cluster_label)

    def is_labeled(self):
        return self.cluster_label != -1


class CentroidPoint(Point):
    """Centroid point used by k-means"""
    def __init__(self, dimension, coords=None):
        super().__init__(dimension, coords)

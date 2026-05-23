import os
from point import *

class Dataset:
    """
    Contains input parameters n, d, k as integers
    as well as a list of ClusterPoints that can be modified by k-means
    """
    def __init__(self, n, d, k, points):
        self.n = n
        self.d = d
        self.k = k
        self.cluster_points = points

    def score(self, centroids):
        sum = 0
        for p in self.cluster_points:
            assert p.is_labeled(), "Point p=({}) is not labelled.".format(p)
            sum = sum + p.sq_distance_to(centroids[p.cluster_label])
        return sum

    def avg_score(self, centroids):
        return self.score(centroids) / len(self.cluster_points)

    def print_output(self, centroids):
        print("{} {} {}".format(self.n, self.d, len(centroids)))
        # print cluster points
        for point in self.cluster_points:
            print(point)
        # print centroid points
        for point in centroids:
            print(point)

    def write_output(self, centroids, path_out, assignment_nr):
        # create directory if not exists
        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        with open(path_out, "w+") as f:
            f.write("{}\n".format(assignment_nr))
            f.write("{} {} {}\n".format(self.n, self.d, len(centroids)))

            # print cluster points
            for point in self.cluster_points:
                f.write(str(point) + "\n")
            # print centroid points
            for point in centroids:
                f.write(str(point) + "\n")
          
    @staticmethod
    def read_input(file, is_labelled=False):
        """
        Parses the input file into a Dataset object

        :param file:    opened file object (or stdio)
        :return         a Dataset object for the input set
        """
        # read first line and split into list
        words = file.readline().split()

        # check whether first line contains correct nr. of parameters
        assert len(words) == 3

        n, d, k = int(words[0]), int(words[1]), int(words[2])

        # Read n points
        points = []
        for _ in range(n):
            words = file.readline().split()

            # check whether input line contains exactly d coordinates
            label = -1
            if not is_labelled:
                assert len(words) == d
            else:
                assert len(words) == d + 1
                label = int(words[-1])

            # Read d coords
            coords = [float(word) for word in words[:d]]

            # Add point to array
            points.append(ClusterPoint(d, coords, label))

        return Dataset(n, d, k, points)

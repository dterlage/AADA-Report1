import numpy as np
from point import *

def create_centroid(point):
    return CentroidPoint(point.dimension, list(point.coords))


def init_Gonzalez(input_obj):
    points = input_obj.cluster_points
    centroids = [create_centroid(points[0])]
    min_dist = [p.sq_distance_to(centroids[0]) for p in points]

    while len(centroids) < input_obj.k:
        # Select furthest point as new centroid, and calculate new minimum furthest distance
        new_centroid = create_centroid(points[np.argmax(min_dist)])
        centroids.append(new_centroid)

        new_dist = [p.sq_distance_to(new_centroid) for p in points]
        min_dist = np.minimum(min_dist, new_dist)

    return centroids



def init_kmeans_plusplus(input_obj):
    """
    Calculates the initial centroid placement

    :param input_obj:   the input object
    :return:            a list of k centroids in the plane
    """
    rng = np.random.default_rng()

    new_centroid = input_obj.cluster_points[rng.integers(input_obj.n)]
    centroids = [create_centroid(new_centroid)]

    while len(centroids) < input_obj.k:
        distance = np.array([
            min(p.sq_distance_to(c) for c in centroids)
            for p in input_obj.cluster_points
        ])

        probability = distance / np.sum(distance)
        new_centroid = rng.choice(input_obj.cluster_points, p=probability)
        centroids.append(create_centroid(new_centroid))

    return centroids



def cluster(input_obj, init):
    """
    Perform k-means clustering on the input set

    :param input_obj:   the input object
    :param init:        the function with input_obj within calculating the initial centroids
    :return:            a list of k centroids in the plane and the iteration count
    """
    centroids = init
    count_i = 0     # Keeps track of the number of iterations it takes from the initialization till the final centroids are found

    change = True
    while change:
        change = False
        count_i += 1      

        for p in input_obj.cluster_points:
            near_cluster = int(np.argmin([p.sq_distance_to(c) for c in centroids]))
            if p.cluster_label != near_cluster:
                p.cluster_label = near_cluster
                change = True

        for i, centroid in enumerate(centroids):
            assigned_points = [p for p in input_obj.cluster_points if p.cluster_label == i]
            if assigned_points:
                centroid.coords = [0.0] * centroid.dimension
                for p in assigned_points:
                    centroid.add(p)
                centroid.div(len(assigned_points))

    return centroids, count_i



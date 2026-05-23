import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def write_input_file(path: str, data, k:int):
    n, d = data.shape

    with open(path, "w") as f:
        f.write(f"{n} {d} {k}\n")

        for i in range(n):
            coord = " ".join(str(x) for x in data[i])
            f.write(f"{coord}\n")


def generate_data(n_sample:int,
                  k_centers:int,
                  n_outliers:int,
                  d:int,
                  std:float = 1.5,
                  cluster_box:float = 15.0,
                  dist_outliers:float = 3.0,
                  random_state:int = 42):
    """
    :param n_sample: number of total samples (incl. outliers)
    :param k_centers: number of centers to generate main clusters around
    :param n_outliers: number of outliers to generate
    :param d: dimensions
    :param std: standard deviation of generated clusters
    :param cluster_box: Size in which clusters and points can be created
    #:param outlier_range: range of outliers (square around (0,0))
    :param dist_outliers: Multiplied by std of generated clusters to provide minimum outlier distance

    :return: np.array of shape (n_sample, d)
    """

    if n_sample < n_outliers:
        raise Exception('More outliers than total sample size')


    X_clusters, _, centers = make_blobs(
        n_samples=n_sample - n_outliers, centers=k_centers,
        n_features=d, cluster_std=std,
        center_box=(-cluster_box, cluster_box),
        random_state=random_state, return_centers=True
    )

    # Create candidate outliers and compute distance to nearest centroids
    candidates = np.random.uniform(-cluster_box, cluster_box, (n_outliers * 40, d))
    distances = np.min(np.linalg.norm(
        candidates[:, np.newaxis, :] - centers[np.newaxis, :, :],  # (n_candidates, k_centers, d)
        axis=2
    ), axis=1)  # shape: (n_candidates,)

    # Outliers must be at least dist_outliers standard deviations away from original centroids
    valid_canidates = candidates[distances >= std * dist_outliers]
    sorted_outliers = np.random.choice(len(valid_canidates), size=n_outliers, replace=False)
    X_outliers = valid_canidates[sorted_outliers]

    # Merge and shuffle created data_in and outliers
    data = np.vstack((X_clusters, X_outliers))
    np.random.shuffle(data)

    return data







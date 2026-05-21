import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

class Dataset:
    def __init__(self, n_sample:int, n_centers:int, n_outliers:int, d:int, std=1.5, outlier_range=20):
        self.n = n_sample + n_outliers
        self.center = n_centers
        self.out = n_outliers
        self.d = d
        self.std = std
        self.outlier_range = outlier_range


        # Generate artificial clusters and outliers
        X_clusters, _, centers = make_blobs(
            n_samples=n_sample, centers=self.center,
            n_features=self.d, cluster_std=self.std,
            center_box=(-0.5 * outlier_range, 0.5 * outlier_range), # Allow to create points in half the region that the outliers can use (Could be altered during experiments)
            random_state=42, return_centers=True
        )

        # Create candidate outliers and compute distance to nearest centroids
        candidates = np.random.uniform(-self.outlier_range, self.outlier_range, (self.out * 20, self.d))
        distances = np.min(np.linalg.norm(
            candidates[:, np.newaxis, :] - centers[np.newaxis, :, :],  # (n_candidates, n_centers, d)
            axis=2
        ), axis=1)  # shape: (n_candidates,)

        # Outliers must be at least 3 standard deviations away from original centroids (Could be altered during experiments)
        valid_canidates = candidates[distances > self.std * 3]
        sorted_outliers = np.random.choice(len(valid_canidates), size=n_outliers, replace=False)
        X_outliers = valid_canidates[sorted_outliers]

        # Merge and shuffle created data and outliers
        data = np.vstack((X_clusters, X_outliers))
        np.random.shuffle(data)
        self.data = data


    def visualize(self):
        if self.d < 3:
            data = self.data
            plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.5)
            plt.title(f"{self.d}D Dataset with Outliers")
            plt.show()
        else:
            print('Too many dimensions')


    def save_csv(self, filename):
        data = pd.DataFrame(self.data)
        data.to_csv(f'data/{filename}-{self.d}D.csv', index=False, header=False)


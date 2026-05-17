import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

class Dataset:
    def __init__(self, n_sample, n_centers, n_outliers, d):
        self.n = n_sample
        self.center = n_centers
        self.out = n_outliers
        self.d = d


    def generate_data(self):
        X_clusters, y_clusters = make_blobs(n_samples=self.n, centers=self.center, n_features=self.d, cluster_std=1.5, random_state=42)
        X_outliers = np.random.uniform(low=-20, high=20, size=(self.out, self.d))
        data  = np.vstack((X_clusters, X_outliers))
        self.data=data
        return data

    def visualize(self):
        data = self.data
        plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.5)
        plt.title(f"{self.d}D Dataset with Outliers")
        plt.show()

    def save_csv(self, filename):
        data = pd.DataFrame(self.data)
        data.to_csv(f'data/{filename}-{self.d}D.csv', index=False, header=False)


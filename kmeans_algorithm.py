from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset


def Gonzalez(dataset:Dataset, k:int):
    points = dataset.data
    centroids = [points[1]]

    min_dist = np.linalg.norm(points - centroids[0], axis=1)

    while len(centroids) < k:
        # Select furthest point as new centroid, and calculate new minimum furthest distance
        centroids.append(points[np.argmax(min_dist)])

        new_dist = np.linalg.norm(points - centroids[-1], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return centroids


def compare_plots(dataset:Dataset, kmeans, gonzalez):
    points = dataset.data
    models = [
        (kmeans,    f"KMeans++ Initialization | Cost = {round(kmeans.inertia_, 2)}"),
        (gonzalez,  f"Gonzalez Initialization | Cost = {round(gonzalez.inertia_, 2)}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model, title) in zip(axes, models):
        labels    = model.labels_
        centroids = model.cluster_centers_

        scatter = ax.scatter(points[:, 0], points[:, 1],
                             c=labels, cmap='viridis',
                             alpha=0.6, edgecolors='k', linewidths=0.3)

        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='X', s=200,
                   label='Centroids', edgecolors='black', linewidths=1.5, zorder=5)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

    plt.tight_layout()
    plt.show()



generated_data = Dataset(200, 8, 300, 2, 2, 40)
kmeans_gonzalez = KMeans(n_clusters=3, init=Gonzalez(generated_data, 3), random_state=42).fit(generated_data.data)
kmeans_kplus = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(generated_data.data)
compare_plots(generated_data, kmeans_kplus, kmeans_gonzalez)

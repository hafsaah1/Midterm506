import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, init_method="random"):
        self.k = k
        self.max_iters = max_iters
        self.init_method = init_method
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        if self.init_method == "random":
            self.centroids = self.initialize_random(X)
        elif self.init_method == "farthest_first":
            self.centroids = self.initialize_farthest_first(X)
        elif self.init_method == "kmeans++":
            self.centroids = self.initialize_kmeans_plus_plus(X)
        else:
            raise ValueError("Invalid initialization method")

        for _ in range(self.max_iters):
            self.clusters = self.create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.calculate_new_centroids(X)

            if np.all(old_centroids == self.centroids):
                break

    def initialize_random(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for point in X:
            closest_centroid = np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids])
            clusters[closest_centroid].append(point)
        return clusters

    def calculate_new_centroids(self, X):
        return np.array([np.mean(cluster, axis=0) for cluster in self.clusters])

    def predict(self, X):
        return np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids]) for point in X])

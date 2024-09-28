from flask import Flask, render_template, request
import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_kmeans", methods=["POST"])
def run_kmeans():
    k = int(request.form["k"])
    init_method = request.form["init_method"]

    # Generate random dataset
    X = np.random.rand(300, 2) * 20 - 10

    # Initialize and run KMeans
    kmeans = KMeans(k=k, init_method=init_method)
    kmeans.fit(X)

    # Plot the clusters and centroids
    plot_url = plot_clusters(kmeans.clusters, kmeans.centroids)

    return render_template("index.html", plot_url=plot_url)

def plot_clusters(clusters, centroids):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    plt.figure()
    for idx, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[idx % len(colors)], label=f'Cluster {idx+1}')
    
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100, label='Centroids')

    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{plot_url}"

if __name__ == "__main__":
    app.run(debug=True, port=3000)

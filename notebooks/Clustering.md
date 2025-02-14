---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: cyto_ml
    language: python
    name: python3
---

Notebook examination of applying unsupervised clustering methods to vector embeddings and visualising the results

Keywords: K-means, DBScan, T-SNE, other?

Paper reference for approach: https://aslopubs.onlinelibrary.wiley.com/doi/full/10.1002/lno.12101#lno12101-sec-0025-title

Possibly interesting if we try the transformer-based plankton model from Turing: https://link.springer.com/chapter/10.1007/978-3-030-74251-5_23 


```python
import sys
sys.path.append('../')
from cyto_ml.data.vectorstore import vector_store
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
```

Load our embeddings into a form suitable for throwing at clustering algorithms, 2048 features might be optimistic and we need to first reduce them!

```python
store = vector_store('sqlite','/home/jowals/plankton_ml/data/untagged-images-lana.db')
res = store.embeddings()
ids = store.ids()
X = np.array(res)
```

This doesn't work with such a high number of features even with `make_blobs` generating pre-clustered data with 2048 features, and tips like scaling values.

So either PCA first, or we just stick with K-means as a simpler effort and work our way back here.

```python
def do_dbscan(X):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    db = DBSCAN(eps=0.7, min_samples=100).fit(X_scaled)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

do_dbscan(X)
```

Sense check on generated dataset with same number of features and three natural clusters - uncomment to see it

```python
#from sklearn.datasets import make_blobs
#X, y = make_blobs(n_samples=1000, centers=3, n_features=512, random_state=0)
#do_dbscan(X)
```

Fall back to a K-means approach, just to try and get some visual feedback

```python
# Set the number of clusters
num_clusters = 5  # Adjust based on your data

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
```

```python

clusters = dict(zip(set(labels), [[] for _ in range(len(set(labels)))]))

for index, id in enumerate(ids):
    l = labels[index]
    clusters[l].append(id)
```

```python
i = 0 # picked at random
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 12),  # creates 2x2 grid of axes
                 axes_pad=0.2,  # pad between axes in inch.
                 )

for index, ax in enumerate(grid):
    # Iterating over the grid returns the Axes.
    ax.imshow(io.imread(clusters[i][index]))

```

To be continued

* Iteration with cluster sizes - 10 was picked arbitrarily, 1 and 2 look like detritus
* Proper look at image quality - what's getting lost between the FlowCam and here
* Nicer way of doing this than a notebook, that has some level of reuse value for other image projects


```python
%pip install umap-learn hdbscan
```

Try dimensionality reduction 

```python
from umap import UMAP
from hdbscan import HDBSCAN
```

Basic usage from https://umap-learn.readthedocs.io/en/latest/basic_usage.html "by default, UMAP reduces down to 2D"

```python
reducer = UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
embedding = reducer.fit_transform(res)
embedding.shape
```

```python
X = embedding
```

Silhoutte analysis as per https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

```python
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

range_n_clusters = [3, 5, 7, 8, 10, 12, 15]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[2], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 3nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()
```

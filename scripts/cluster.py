"""Tiny DVC stage that trains and saves a K-means model from embeddings"""

import pickle
import os
import yaml

from sklearn.cluster import KMeans
from cyto_ml.data.vectorstore import embeddings, vector_store


def main() -> None:
    # You can supply -p params to dvc as an alternative to params.yaml
    # But this (from the example) suggests they don't get overriden?
    params = yaml.safe_load(open("params.yaml"))["cluster"]
    kmeans = KMeans(n_clusters=params["n_clusters"], random_state=42)
    store = vector_store("plankton")
    X = embeddings(store)
    kmeans.fit(X)


    # We supply a -o for output directory - sure this writes there?
    # The examples show the path hard-coded in the script, too
    # https://dvc.org/doc/start/data-pipelines/data-pipelines
    # Keeps failing at output does not exist, deletes it if i create it first!
    
    with open("kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)




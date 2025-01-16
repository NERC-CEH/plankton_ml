"""Tiny DVC stage that trains and saves a K-means model from embeddings"""

import logging
import pickle
import os
import yaml

from sklearn.cluster import KMeans
from cyto_ml.data.vectorstore import vector_store


def main() -> None:

    os.makedirs("../models", exist_ok=True)

    # You can supply -p params to dvc as an alternative to params.yaml
    # But this (from the example) suggests they don't get overriden?
    params = yaml.safe_load(open("params.yaml"))
    collection_name = params.get("collection", "untagged-images-lana")
    try:
        stage_params = params["cluster"]
        n_clusters = stage_params["n_clusters"]
    except:
        logging.info("No parameters for stage found - default to 5 clusters")
        n_clusters = 5

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    store = vector_store("sqlite", collection_name)
    X = store.embeddings()
    kmeans.fit(X)

    # We supply a -o for output directory - this doesn't ensure we write there.
    # The examples show the path hard-coded in the script, too
    # https://dvc.org/doc/start/data-pipelines/data-pipelines
    # Output directory will be deleted at the start of the stage;
    # It's the script's responsibility to ensure it's recreated

    with open(f"../models/kmeans-{collection_name}.pkl", "wb") as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()

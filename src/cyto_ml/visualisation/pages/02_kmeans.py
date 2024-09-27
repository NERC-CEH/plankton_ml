import logging

import streamlit as st
from sklearn.cluster import KMeans

from cyto_ml.visualisation.app import (
    cached_image,
    image_embeddings,
    image_ids,
)

logging.basicConfig(level=logging.INFO)

DEPTH = 8


@st.cache_resource
def kmeans_cluster() -> KMeans:
    """
    K-means cluster the embeddings, option in session for default size

    """
    X = image_embeddings()
    n_clusters = st.session_state["n_clusters"]
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


@st.cache_data
def image_labels() -> dict:
    """
    TODO good form to move all this into cyto_ml, call from there?
    """
    km = kmeans_cluster()
    clusters = dict(zip(set(km.labels_), [[] for _ in range(len(set(km.labels_)))]))

    for index, _id in enumerate(image_ids()):
        label = km.labels_[index]
        clusters[label].append(_id)
    return clusters


def add_more() -> None:
    st.session_state["depth"] += DEPTH


def do_less() -> None:
    st.session_state["depth"] -= DEPTH


def find_closest() -> None:
    fitted = image_labels()
    closest = fitted[st.session_state["cluster"]]
    st.session_state["closest"] = closest


def show_cluster() -> None:
    # TODO figure out why this renders twice

    for _ in range(0, st.session_state["depth"]):
        cols = st.columns(DEPTH)

        for c in cols:
            c.empty()
            try:
                next_image = st.session_state["closest"].pop()
            except IndexError:
                break
            c.image(cached_image(next_image), width=60)


# TODO some visualisation, actual content, etc
def main() -> None:
    # start with this cluster label
    if "cluster" not in st.session_state:
        st.session_state["cluster"] = 1

    # start kmeans with this many target clusters
    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 5

    # show this many images * 8 across
    if "depth" not in st.session_state:
        st.session_state["depth"] = 8

    st.selectbox(
        "cluster label",
        [x for x in range(0, st.session_state["n_clusters"])],
        key="cluster",
        on_change=find_closest,
    )

    st.selectbox(
        "number of clusters",
        [3, 5, 8],
        key="n_clusters",
        on_change=kmeans_cluster,
    )

    st.button("more", on_click=add_more)

    st.button("less", on_click=do_less)

    find_closest()
    show_cluster()


if __name__ == "__main__":
    main()

"""
Streamlit application to visualise how plankton cluster
based on their embeddings from a deep learning model

* Metadata in intake catalogue (basically a dataframe of filenames
  - later this could have lon/lat, date, depth read from Exif headers
* Embeddings in chromadb, linked by filename

"""

import random
from io import BytesIO
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from cyto_ml.data.image import normalise_flowlr
from cyto_ml.data.vectorstore import client, embeddings, vector_store

load_dotenv()


def collections() -> list:
    return [c.name for c in client.list_collections()]


@st.cache_resource
def store(coll: str) -> None:
    """
    Load the vector store with image embeddings.
    TODO switch between different collections, not set in .env
    Set as "EMBEDDINGS" in .env or defaults to "plankton"
    """
    return vector_store(coll)


@st.cache_data
def image_ids(coll: str) -> list:
    """
    Retrieve image embeddings from chroma database.
    TODO Revisit our available metadata
    """
    result = store(coll).get()
    return result["ids"]


@st.cache_data
def image_embeddings() -> list:
    return embeddings(store(st.session_state["collection"]))


def closest_n(url: str, n: Optional[int] = 26) -> list:
    """
    Given an image URL return the N closest ones by cosine distance
    """
    s = store(st.session_state["collection"])
    embed = s.get([url], include=["embeddings"])["embeddings"]
    results = s.query(query_embeddings=embed, n_results=n)
    return results["ids"][0]  # by index because API assumes query always multiple


@st.cache_data
def cached_image(url: str) -> Image:
    """
    Read an image URL from s3 and return a PIL Image
    Hopefully caches this per-image, so it'll speed up
    We tried streamlit_clickable_images but no tiff support
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # Special handling for Flow Cytometer images,
    # All 16 bit greyscale in low range of values
    if image.mode == "I;16":
        image = normalise_flowlr(image)

    return image


def closest_grid(start_url: str, size: Optional[int] = 65) -> None:
    """
    Given an image URL, render a grid of the N nearest images
    by cosine distance between embeddings
    N defaults to 26
    """
    closest = closest_n(start_url, size)

    # TODO understand where layout should happen
    rows = []
    for _ in range(0, 8):
        rows.append(st.columns(8))

    for index, _ in enumerate(rows):
        for c in rows[index]:
            try:
                next_image = closest.pop()
            except IndexError:
                break
            c.image(cached_image(next_image), width=60)
            c.button("this", key=next_image, on_click=pick_image, args=[next_image])


def create_figure(df: pd.DataFrame) -> go.Figure:
    """
    Creates scatter plot based on handed data frame
    TODO replace this layout with
    a) most basic image grid, switch between clusters
    b) ...
    """
    color_dict = {i: px.colors.qualitative.Alphabet[i] for i in range(0, 20)}
    color_dict[-1] = "#ABABAB"
    topic_color = df["topic_number"].map(color_dict)
    fig = go.Figure(
        data=go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker_color=topic_color,
            customdata=df["doc_id"],
            text=df["short_title"],
            hovertemplate="<b>%{text}</b>",
        )
    )
    fig.update_layout(height=600)
    return fig


def random_image() -> str:
    ids = image_ids(st.session_state["collection"])
    # starting image
    test_image_url = random.choice(ids)
    return test_image_url


def pick_image(image: str) -> None:
    st.session_state["random_img"] = image


def show_random_image() -> None:
    if st.session_state["random_img"]:
        st.image(cached_image(st.session_state["random_img"]))
        st.write(st.session_state["random_img"])


def main() -> None:
    """
    Main method that sets up the streamlit app and builds the visualisation.
    """

    if "random_img" not in st.session_state:
        st.session_state["random_img"] = None

    colls = collections()
    if "collection" not in st.session_state:
        st.session_state["collection"] = colls[0]

    st.set_page_config(layout="wide", page_title="Plankton image embeddings")

    st.title("Image embeddings")
    st.write(f"{len(image_ids(st.session_state['collection']))} images in {st.session_state["collection"]}")
    # the generated HTML is not lovely at all

    st.selectbox(
        "image collection",
        colls,
        key="collection",
    )

    st.session_state["random_img"] = random_image()
    show_random_image()

    st.text("<-- random image")

    st.button("try again", on_click=random_image)

    # TODO figure out how streamlit is supposed to work
    closest_grid(st.session_state["random_img"])


if __name__ == "__main__":
    main()

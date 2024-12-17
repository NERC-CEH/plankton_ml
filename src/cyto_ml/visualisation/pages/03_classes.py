import logging

import streamlit as st

from cyto_ml.visualisation.app import cached_image, collections, store

logging.basicConfig(level=logging.INFO)

DEPTH = 8


@st.cache_data
def add_more() -> None:
    st.session_state["depth"] += DEPTH


def do_less() -> None:
    st.session_state["depth"] -= DEPTH


def by_class() -> None:
    classed = store(st.session_state["collection"]).labelled(st.session_state["class_label"])
    st.session_state["closest"] = classed


def show_cluster() -> None:
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
    # duplicate logic from main page, how should this state be shared?

    colls = collections()
    if "collection" not in st.session_state:
        st.session_state["collection"] = colls[0]

    classlist = store(st.session_state["collection"]).classes()
    if "class_label" not in st.session_state:
        st.session_state["class_label"] = classlist[0]

    st.selectbox(
        "image collection",
        colls,
        key="collection",
    )

    # show this many images * 8 across
    if "depth" not in st.session_state:
        st.session_state["depth"] = 8

    st.selectbox(
        "class label",
        classlist,
        key="class_label",
        on_change=by_class,
    )

    st.button("more", on_click=add_more)

    st.button("less", on_click=do_less)

    by_class()
    show_cluster()


if __name__ == "__main__":
    main()

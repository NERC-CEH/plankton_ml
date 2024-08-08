# Plankton ML

This repository contains code and configuration for processing and analysing images of plankton samples. It's experimental, serving as much as a proposed template for new projects than as a project in itself.

It's a companion project to an R-shiny based image annotation app that is not yet released, written by researchers and data scientists at the UK Centre for Ecology and Hydrology in the early stages of a collaboration that was placed on hold.

## Installation

### Python environment setup

Use anaconda or miniconda to create a python environment using the included `environment.yml`

```
conda env create -f environment.yml
```

Please note that this is specifically pinned to python 3.9 due to dependency versions; we make experimental use of the [CEFAS plankton model available through SciVision](https://sci.vision/#/model/resnet50-plankton), which in turn uses an older version of pytorch that isn't packaged above python 3.9.

### Object store connection

`.env` contains environment variable names for S3 connection details for the [JASMIN object store](https://github.com/NERC-CEH/object_store_tutorial/). Fill these in with your own credentials. If you're not sure what the `ENDPOINT` should be, please reach out to one of the project contributors listed below. 


### Package installation

Get started by cloning this repository and running

`pip install -e .`

### Running tests

`python -m pytest` or `py.test`

## Contents

### Catalogue creation

`scripts/intake_metadata.py` is a proof of concept that creates a configuration file for an [intake](https://intake.readthedocs.io/en/latest/) catalogue - a utility to make reading analytical datasets into analysis workflows more reproducible and less effortful.

### Feature extraction

Experiment testing workflows by using [this plankton model from SciVision](https://sci.vision/#/model/resnet50-plankton) to extract features from images for use in similarity search, clustering, etc.

### Running Jupyter notebooks

The `notebooks/` directory contains Markdown (`.md`) representations of the notebooks.
To create Jupyter notebooks (`.ipynb`), run the following command with the conda environment activated:

```sh
jupytext --sync notebooks/*
```

If you modify the contents of a notebook, run the command after closing the notebook to re-sync the `.ipynb` and `.md` representations before committing.

For more information see the [Jupytext docs](https://jupytext.readthedocs.io/en/latest/).

## Visualisation

Streamlit app based off the [text embeddings for EIDC catalogue metadata](https://github.com/NERC-CEH/embeddings_app/) one

```
streamlit run cyto_ml/visualisation/visualisation_app.py
```

The demo should automatically open in your browser when you run streamlit. If it does not, connect using: http://localhost:8501.

### TBC (object store upload, derived classifiers, etc)


## Contributors

[Jo Walsh](https://github.com/metazool/)
[Alba Gomez Segura](https://github.com/albags)
[Ezra Kitson](http://github.com/Kzra)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)


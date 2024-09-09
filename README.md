# Plankton ML

This repository contains code and configuration for processing and analysing images of plankton samples. It's experimental, serving as much as a proposed template for new projects than as a project in itself.

It's a companion project to an R-shiny based image annotation app that is not yet released, written by researchers and data scientists at the UK Centre for Ecology and Hydrology in the early stages of a collaboration that was placed on hold.

## Installation

### Environment and package installation

#### Using pip

Create a fresh virtual environment in the repository root using Python >=3.12 and (e.g.) `venv`: 

```
python -m venv venv
```

Next, install the package using `pip`:

```
python -m pip install .
```

Most likely you are interested in developing and/or experimenting, so you will probably want to install the package in 'editable' mode (`-e`), along with dev tools and jupyter notebook functionality

```
python -m pip install -e .[all]
```

#### Using conda

Use anaconda or miniconda to create a python environment using the included `environment.yml`

```
conda env create -f environment.yml
conda activate cyto_ml
```

Next install this package _without dependencies_:

```
python -m pip install --no-deps -e .
```

#### exiftool

We use `exiftool` to write basic metadata (latitude/longitude of observation, plus timestamp) into individual plankton images extracted from the larger "collage" format that the FlowCam microscope exports them in.

[Guidance for installing exiftool](https://www.geeksforgeeks.org/installing-and-using-exiftool-on-linux/)

Ubuntu: `sudo apt install libimage-exiftool-perl`
Centos: `sudo yum install libimage-exiftool-perl`
Or in an environment without root access:
```
git clone https://github.com/exiftool/exiftool.git
export PATH=$PATH:exiftool
```
 
### Object store connection

`.env` contains environment variable names for S3 connection details for the [JASMIN object store](https://github.com/NERC-CEH/object_store_tutorial/). Fill these in with your own credentials. If you're not sure what the `ENDPOINT` should be, please reach out to one of the project contributors listed below. 

### Running tests

`pytest` or `py.test`

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


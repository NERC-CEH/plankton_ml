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
 
## Object store 

## Connection details

`.env` contains environment variable names for S3 connection details for the [JASMIN object store](https://github.com/NERC-CEH/object_store_tutorial/). Fill these in with your own credentials. If you're not sure what the `AWS_URL_ENDPOINT` should be, please reach out to one of the project contributors listed below. 

## Object store API

The [object_store_api](https://github.com/NERC-CEH/object_store_api) project provides a web-based API to help manage your image data, for use with JASMIN's s3 store.

Please [see its documentation](https://github.com/NERC-CEH/object_store_api) for different modes of running the API. The simplest, for single user / testing purposes is:

`python src/os_api/api.py`

## Feature extraction API

FastAPI wrapper around different models - POST an image URL, get back embeddings 

## Label Studio ML backend

Pre-annotation backend for Label Studio following their standard pattern.

Build an image embedding model which will assign a likely-detritus tag:

```
cd scripts
dvc repro
```

Application is in `src/label_studio_cyto_ml`

[Setup documentation](src/label_studio_cyto_ml/README.md)

Short version, for testing

```
cd src
label-studio-ml start ./label_studio_cyto_ml
```

## Pipelines

### DVC 

Please see [DVC.yaml] for notes and walkthroughs on different ways of using [Data Version Control](https://dvc.org/) both to manage data within a git repository, and to manage sets of scripts as a reproducble pipeline with minimal intervention.

This _very basic_ setup has several stages - build an index of images in an object store (s3 bucket), extract and store their embeddings using a pre-trained neural network, and train and save a classifier based on the embeddings.

`cd scripts`
`dvc repro`

## Luigi

Please see [PIPELINES.md](PIPELINES) for detailed documentation about a pipeline that slices up images exported from a FlowCam instrument, adds spatial and temporal metadata into their EXIF headers based on a directory naming convention agreed with researchers, and uploads them to object storage.


### Running tests

`pytest` or `py.test`

## Contents

### Feature extraction

The repository contains work on _feature extraction_ from different off-the-shelf ML models that have been trained on datasets of plankton imagery.

The approach is useful for image search, clustering based on image similarity, and potentially for timeseries analysis of features given an image collection that forms a timeseries.

* [ResNet50 plankton model from SciVision](https://sci.vision/#/model/resnet50-plankton)
* [ResNet18 plankton model from Alan Turing Inst]

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
streamlit run src/cyto_ml/visualisation/app.py
```

The demo should automatically open in your browser when you run streamlit. If it does not, connect using: http://localhost:8501.





DAG / pipeline elements 

## Contributors

[Jo Walsh](https://github.com/metazool/)
[Alba Gomez Segura](https://github.com/albags)
[Ezra Kitson](http://github.com/Kzra)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)


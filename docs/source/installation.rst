Installation
===========

Using pip
---------

Create a fresh virtual environment using Python >=3.12::

    python -m venv venv
    source venv/bin/activate
    pip install -e .[all]

Using conda
----------

Create environment from included configuration::

    conda env create -f environment.yml
    conda activate cyto_ml
    pip install --no-deps -e .
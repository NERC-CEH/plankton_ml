Label Studio
============

We provide a set for the Label Studio project (data labelling for machine learning) with an ML Backend that suggests, at a minimum, whether or not an image is plankton or detritus.

There's an example `docker-compose.yml` file in this project for building and running Label Studio and its backend.

The full configuration for the running service is in `this private project <https://github.com/ukceh-rse/podman-host>` - please contact a member of the RSE group if you would like access.

ML Backend notes
----------------

`src/label_studio_cyto_ml/model.py` contains our custom model code.

It runs two models

* A ResNet (could be any deep learning model) that extracts embeddings from an image
* A kmeans clustering model which fits the resulting embeddings with a specific label_studio_cyto_ml

Return format 
-------------

It took a bit of figuring out to get the return format right. A single prediction needs returned as an array of results, like this: 

`PredictionValue(result=[{"id": int(label), "text": "test", "type": "Choices"}])`

The `ModelResponse` is then an array of these `PredictionValue` objects

`ModelResponse(predictions=predictions)`

The prediction needs a `type` value which internally is a `control tag <https://labelstud.io/tags/choices>` - many types of these for different media, our checkbox / radio buttons are `Choices`

The input to the annotation task looks like this (defined when setting up the project)

{'organism_type': {'type': 'Choices', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['Not-plankton', 'Plankton', 'Debris'], 'labels_attrs': {'Not-plankton': {'value': 'Not-plankton'}, 'Plankton': {'value': 'Plankton'}, 'Debris': {'value': 'Debris'}}}, 'morphology': {'type': 'Choices', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['Mucilage', 'Flagella', 'Cilia', 'Aerotopes', 'Akinetes', 'Heterocytes', 'Theca/test/exoskeletal structures', 'Eggs', 'Ephippia'], 'labels_attrs': {'Mucilage': {'value': 'Mucilage'}, 'Flagella': {'value': 'Flagella'}, 'Cilia': {'value': 'Cilia'}, 'Aerotopes': {'value': 'Aerotopes'}, 'Akinetes': {'value': 'Akinetes'}, 'Heterocytes': {'value': 'Heterocytes'}, 'Theca/test/exoskeletal structures': {'value': 'Theca/test/exoskeletal structures'}, 'Eggs': {'value': 'Eggs'}, 'Ephippia': {'value': 'Ephippia'}}}, 'life_form': {'type': 'Choices', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['Unicellular', 'Colony', 'Filament'], 'labels_attrs': {'Unicellular': {'value': 'Unicellular'}, 'Colony': {'value': 'Colony'}, 'Filament': {'value': 'Filament'}}}, 'shape': {'type': 'Choices', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': ['Spiky', 'Round', 'Rod-like'], 'labels_attrs': {'Spiky': {'value': 'Spiky'}, 'Round': {'value': 'Round'}, 'Rod-like': {'value': 'Rod-like'}}}, 'ta': {'type': 'TextArea', 'to_name': ['image'], 'inputs': [{'type': 'Image', 'valueType': None, 'value': 'image'}], 'labels': [], 'labels_attrs': {}}}
::

`Troubleshooting pre-annotations <https://labelstud.io/guide/troubleshooting#Pre-annotations>`

Connection to Label Studio
--------------------------

Each Label Studio project needs configured to use an ML backend service.

This could be our custom one or a range of off-the-shelf options (like SAM for segmentation)


* Navigate to `Project/Settings/Model`
* Add the URL referring to the container by name, as it reads in `docker-compose.yml`

For example, our `docker-compose.yml` has three services, one is named `ml-backend`, so this is the URL that goes in the project settings:

`http://ml-backend:9090/`

Label Studio analytics
----------------------

We've had some issues with Label Studio enabling analytics by default, then page loads stalling because the analytics service is throttling requests.

As of writing this needs a build from source as well as configuration options, but should be fixed when version 1.17.1 becomes the default docker build (see `this issue https://github.com/HumanSignal/label-studio/issues/6430`)

git clone https://github.com/HumanSignal/label-studio.git
podman build -t heartexlabs/label-studio:latest .
::



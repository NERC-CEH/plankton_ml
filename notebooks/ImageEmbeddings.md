---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Use this with the `cyto_39` environment (the scivision model needs a specific version of `pytorch` that isn't packaged for >3.9, i have raised a Github issue asking if they plan to update it)

`conda env create -f environment.yml`
`conda activate cyto_39`

```python
import os
from scivision import load_pretrained_model, load_dataset
from dotenv import load_dotenv
import torch
import torchvision
import chromadb
import sys
sys.path.append('../')
from cyto_ml.models.scivision import prepare_image
from intake_xarray import ImageSource
load_dotenv()  # sets our object store endpoint and credentials from the .env file
```

```python
dataset = load_dataset(f"{os.environ.get('ENDPOINT', '')}/metadata/intake.yml")
model = load_pretrained_model("https://github.com/alan-turing-institute/plankton-cefas-scivision")
dataset.test_image().to_dask()
```

The scivision wrapper depends on this being an xarray Dataset with settable attributes, rather than a DataArray

Setting exif_tags: True (Dataset) or False (DataArray) is what controls this
https://docs.xarray.dev/en/stable/generated/xarray.DataArray.to_dataset.html 

https://github.com/alan-turing-institute/scivision/blob/07fb74e5231bc1d56cf39df38c19ef40e3265e4c/src/scivision/io/reader.py#L183
https://github.com/intake/intake/blob/29c8878aa7bf6e93185e2c9639f8739445dff22b/intake/__init__.py#L101

But now we're dependent on image height and width metadata being set in the EXIF tags to use the `predict` interface, this is set in the model description through `scivision`, this is brittle

https://github.com/alan-turing-institute/plankton-cefas-scivision/blob/main/resnet50_cefas/model.py#L71



A quick look at the example dataset that comes with the model, for reference


In this case we don't want to use the `predict` interface anyway (one of N class labels) - we want the features that go into the last fully-connected layer (as described here https://stackoverflow.com/a/52548419)

```python
network = torch.nn.Sequential(*(list(model._plumbing.model.pretrained_model.children())[:-1]))
```

```python
imgs = dataset.test_image().to_dask()
i= imgs.to_numpy()
i.shape

```

https://github.com/alan-turing-institute/plankton-cefas-scivision/blob/main/resnet50_cefas/data.py 



Pass the image through our truncated network and get some embeddings out

```python
o = torch.stack([torchvision.transforms.ToTensor()(i)])
feats = network(o)
feats.shape
```

```python
embeddings = list(feats[0].squeeze(1).squeeze(1).detach().numpy().astype(float))
```

```python
embeddings
```

```python
print(set([type(x) for x in embeddings]))
```

There's been some guesswork up to this point (honestly expected the input would be resized, and concerned it's not normalised) but we have a set of embeddings, at which point pop them in a vector database and see if similarity queries yield any kind of meaningful results, iterate from there

https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/ - langchain has a few, Milvus would be another option
https://python.langchain.com/v0.1/docs/integrations/vectorstores/ - there are so many integrations!
https://www.trychroma.com/ - this looks straightforward to approach

```python
client = chromadb.PersistentClient(path="./vectors")
collection = client.create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )
```

```python
collection.add(
    documents=["test_image"],
    embeddings=[embeddings],
    metadatas=[{"useful": "maybe"}],
    ids=["id2"]  # must be unique, are they required?
)
```

```python
collection.get('id2',include=["embeddings"])
```

```python
index = dataset.plankton().to_dask().compute()
```

```python
index

```

```python
def flat_embeddings(features: torch.Tensor):
    return list(features[0].squeeze(1).squeeze(1).detach().numpy().astype(float))
```

```python
def file_embeddings(row):
    image_data = ImageSource(row.Filename).to_dask()
    embeddings = flat_embeddings(network(prepare_image(image_data)))
    collection.add(
        documents=[row.Filename],
        embeddings=[embeddings],
        ids=[row.Filename]  # must be unique, are they required?
    )


```

```python
from intake_xarray import ImageSource
```

Because all the images have slightly different dimensions as they come out of the FlowCam, we can't batch them
Push them through the model one by one and either build a list of `(id, [embeddings])` pairs, or potentially pop them straight into chromadb as we apply the function, which would keep it more dasklike?

This scales ok at 8000 or so images

```python
collection.count()
```

```python
res = index.apply(file_embeddings, axis=1)
```

```python
collection.count()
```

```python
i = ImageSource(index.loc[0].Filename).to_dask()
i
```

```python
network(prepare_image(i))
```

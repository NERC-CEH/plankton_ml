---
jupyter:
  jupytext:
    formats: ipynb,md
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

# Joe experiments with BioCLIP

```python
import logging
import warnings

from PIL import Image
import torch
import open_clip

# Get clues about what's happening upstream
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(action="ignore", category=FutureWarning)
```

<!-- #region -->
## Loading and inspecting the model

I am following the (very minimal) instructions for loading the BioCLIP from the [BioCLIP huggingface page](https://huggingface.co/imageomics/bioclip#how-to-get-started-with-the-model
), and the more generic ones from the [`open_clip` README](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#usage).

The first step is

```python
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
```

I am looking at the logic inside [`open_clip.create_model_and_transforms`](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py#L375) and it's quite messy/confusing and sparsely documented, with defaults silently overriden. Probably best to just treat this as a black box! Most importantly, it returns three things:

1. an instance of [`open_clip.model.CLIP`](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L220) (a subclass of `torch.nn.Module`) with loaded weights
2. an instance of [`torchvision.transforms.transforms.Compose`](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html) that performs the image preprocessing for training
3. another `Compose` for validation
<!-- #endregion -->

```python
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
model.eval()
print(type(model), type(preprocess_train), type(preprocess_val))
```

### Unpacking CLIP

The `forward` pass of `CLIP` transforms both image and text inputs (if both are provided).

#### Images
Images are passed to the `visual` submodule, which is an instance of [`VisionTransformer`](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py#L434) (You can get an idea for what a `VisionTransformer` contains/does from the `repr` below, but see [source code](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py#L434) for more details.)
The outputs of the transformer are then [normalised](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html) (by dividing by the $\ell_2$ norm). 

#### Text
Text goes through the following stages:

1. Token embedding (`token_embedding`)
2. Transformer (`transformer`)
3. Layer norm (`ln_final1`)
4. Normalise (another division by the $\ell_2$ norm, functional so not listed as a module)

```python
model
```

### Preprocessing

The preprocessing for training and validation are slightly different: during training [`torchvision.transforms.RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) is used to crop a random portion of the image and resize it to (224, 224), whereas in validation the image is resized so that the *smaller* edge is 224 pixels, before a center crop is applied to make the resulting image square.

Note that [`torchvision.transforms.CenterCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html#torchvision.transforms.CenterCrop.forward) pads any dimensions that are smaller than the give center size with zeros.

```python
preprocess_train.transforms
```

```python
preprocess_val.transforms
```

**TODO:** demo the preprocessors on our images


### Tokenizer

```python
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
```

## Plankton images

```python
import os
from cyto_ml.models.scivision import prepare_image
from intake_xarray import ImageSource
from dotenv import load_dotenv

load_dotenv()  # sets our object store endpoint and credentials from the .env file

from intake import open_catalog
```

```python
dataset = open_catalog(f"{os.environ.get('ENDPOINT', '')}/metadata/intake.yml")
imgs = dataset.test_image().to_dask()
imgs.to_numpy().shape
```

```python
index_df = dataset.plankton().to_dask().compute()  # intake-xarray is not sensible here - to_dask.compute() seems silly
index_df
```

<!-- #raw -->
image_data = [
    ImageSource(row).to_dask().to_numpy() for row in index_df["Filename"].iloc[:100]  # take first 100, 535 too many!
]
image_data[0].shape
<!-- #endraw -->

```python
from tqdm.autonotebook import tqdm

fnames = index_df["Filename"]

images = []

for row in tqdm(fnames):
    try:
        image = ImageSource(row).to_dask().to_numpy()
    except OSError as e: # I think one of these is a csv not an image!
        print(e)
    else:
        images.append(image)
```

```python
import matplotlib.pyplot as plt

plt.imshow(images[0])
```

```python
from IPython.display import display
print(images[0].shape)
im = Image.fromarray(images[0], "RGB")
display(im)
```

```python
for i in images[10:50]:
    display(Image.fromarray(i, "RGB"))
```

```python
images[0].shape
```

```python
dimensions = [image.shape[:-1] for image in images]
plt.scatter(*zip(*dimensions), s=3)
```

```python
test = preprocess_val(Image.fromarray(images[0], "RGB"))
print(test.shape)
print(test.min(), test.max())
test

testim = test.permute(2, 1, 0).numpy()
print(testim.shape)
im = Image.fromarray(test.permute(1, 2, 0).numpy(), "RGB")
display(im)
```

```python
#resize, crop, totensor, norm = preprocess_val.transforms

test_image = Image.fromarray(images[16], "RGB")

for transform in preprocess_val.transforms:
    print("transform: ", transform)
    test_image = transform(test_image)
    print("new type: ", type(test_image))


    if isinstance(test_image, Image.Image):
        display(test_image)

    elif isinstance(test_image, torch.Tensor):
        plt.imshow(test_image.transpose(2, 0))
        print(test_image.min(), test_image.max())

    else:
        print(type(test_image))
```

```python
from torchvision.transforms.v2.functional import to_image

test_image = Image.fromarray(images[16], "RGB")

poo = to_image(test_image)
poo.min(), poo.max()
```

## Put in the thing

```python
test_image = Image.fromarray(images[16], "RGB")

display(test_image)
```

```python
in_tensor = preprocess_val(test_image)
in_tensor.shape
```

```python
image_features, text_features, scale = model(in_tensor.unsqueeze(0))
```

```python
image_features.shape
```

### One at a time or it cries

```python
from pathlib import Path
from tqdm.autonotebook import tqdm
assert Path("features").is_dir()

for i, image in tqdm(enumerate(images)):
    image_tensor = preprocess_val(Image.fromarray(image, "RGB"))
    out_feature, _, _ = model(image_tensor.unsqueeze(0))
    torch.save(out_feature, f"features/feature_{i}.pt") 
```

```python
features = torch.cat(
    [torch.load(f"features/feature_{i}.pt") for i in range(len(images))]
)
features.shape
```

```python
!pip install scikit-learn
```

```python
from sklearn.manifold import TSNE

t_sne = TSNE()

embedded = t_sne.fit_transform(features.detach().numpy())
```

```python
embedded.shape
```

```python
plt.scatter(*zip(*embedded))
```

```python

```

## Resources

### CLIP

- [GitHub repo](https://github.com/mlfoundations/open_clip/)
- [Class definition](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L220)
- 

### BioCLIP 

- [arXiv paper](https://arxiv.org/pdf/2311.18803)
- [GitHub repo](https://github.com/Imageomics/bioclip)
- [Model card](https://huggingface.co/imageomics/bioclip)
- [Evaluation using t-SNE](https://imageomics.github.io/bioclip/#intrinsic)

### Evaluation

- [Scikit Learn TSNE class](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#tsne)
- [t-SNE from the original author](https://lvdmaaten.github.io/tsne/)
- [Demo about manifold learning](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#)

```python

```

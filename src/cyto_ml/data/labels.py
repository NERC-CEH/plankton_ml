# Labels for re-use of models from Turing Inst
# https://github.com/ukceh-rse/ViT-LASNet/blob/main/test/test.py

# 3 class Resnet labels
# There's an 18 class model, but focused on marine plankton
# What we care about most now is filtering out detritus

RESNET18_LABELS = [
    "copepod",
    "detritus",
    "noncopepod",
]

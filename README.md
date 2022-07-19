# HEnorm

This repository contains the official implementation of [Bias reduction using combined stain normalization and augmentation for AI-based classification of histological images](https://arxiv.org/). This code was tested with python 3.7 and softwares listed in [requirements](requirements.txt). In order to use it, please install required dependencies by running:
```bash
pip install -r requirements.txt
```

## AugmentHE

AugmentHE was implemented using the [albumentations](https://albumentations.ai/) interface. It can be easily used within a PyTorch dataset:
```python
class MyDataset(Dataset):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.aug = StainAugmentor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        augmented_img = self.aug(image=img)["image"]
        return augmented_img
```

## HEnorm

HEnorm was implemented using [PyTorch](https://pytorch.org/) and [fastai's DynamicUnet](https://docs.fast.ai/vision.models.unet.html). To reproduce the papers' model, simply write:
```python
model = Normalizer("cgr_5_32_4")
```

You can then train this model using a reference dataset and AugmentHE.

## Transforms

All set of transforms described in the paper can be found in [transforms.py](transforms.py).
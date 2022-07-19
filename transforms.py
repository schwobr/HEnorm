from typing import List, Tuple

from albumentations import (
    BasicTransform,
    CenterCrop,
    Flip,
    RandomCrop,
    RandomGamma,
    RandomRotate90,
    RGBShift,
    Transpose,
)

from augmentHE import StainAugmentor


def basic(size: int) -> Tuple[List[BasicTransform], List[BasicTransform]]:
    tfms = [
        RandomCrop(size, size),
        RandomRotate90(),
        Flip(),
        Transpose(),
        RandomGamma(p=0.2),
    ]
    val_tfms = [CenterCrop(size, size)]
    return tfms, val_tfms


def aug_classic(size: int) -> Tuple[List[BasicTransform], List[BasicTransform]]:
    tfms = [
        RandomCrop(size, size),
        RandomRotate90(),
        Flip(),
        Transpose(),
        RandomGamma(),
        RGBShift(30, 30, 30),
    ]
    val_tfms = [CenterCrop(size, size)]
    return tfms, val_tfms


def augHE(size: int) -> Tuple[List[BasicTransform], List[BasicTransform]]:
    tfms = [
        RandomCrop(size, size),
        StainAugmentor(),
        RandomRotate90(),
        Flip(),
        Transpose(),
        RandomGamma(),
    ]
    val_tfms = [CenterCrop(size, size)]
    return tfms, val_tfms

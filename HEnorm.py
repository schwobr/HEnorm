from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from color_converter import ColorConverter
from modules import (
    CBR,
    ConvBnRelu,
    DecoderBlock,
    Hooks,
    LastCross,
    PixelShuffleICNR,
    get_sizes,
    group_norm,
    named_leaf_modules,
)


class DynamicUnet(nn.Module):
    """
    Create a Unet architecture with a custome encoder. Implementation taken from fastai.

    Args:
        encoder_name: name of the encoder to use. Can be any model registered in timm.
            a CBR or a CGR (see modules).
        n_classes: number of classes / output channels.
        input_shape: shape of image tensor inputs.
        pretrained: whether to use a pretrained model. Only available for timm encoders.
    """

    def __init__(
        self,
        encoder_name: str,
        n_classes: int = 2,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        pretrained: bool = True,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        if "cbr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=norm_layer)
            cut = -3
        elif "cgr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=group_norm)
            norm_layer = group_norm
            cut = -3
        else:
            encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                norm_layer=norm_layer,
                pretrained_strict=False,
            )
            cut = -2

        self.encoder = nn.Sequential(*(list(encoder.children())[:cut] + [nn.ReLU()]))
        encoder_sizes, idxs = self._register_output_hooks(input_shape=input_shape)
        n_chans = int(encoder_sizes[-1][1])
        middle_conv = nn.Sequential(
            ConvBnRelu(n_chans, n_chans // 2, 3, norm_layer=norm_layer),
            ConvBnRelu(n_chans // 2, n_chans, 3, norm_layer=norm_layer),
        )
        decoder = [middle_conv]
        for k, (idx, hook) in enumerate(zip(idxs[::-1], self.hooks)):
            skip_chans = int(encoder_sizes[idx][1])
            final_div = k != len(idxs) - 1
            decoder.append(
                DecoderBlock(
                    n_chans,
                    skip_chans,
                    hook,
                    final_div=final_div,
                    norm_layer=norm_layer,
                )
            )
            n_chans = n_chans // 2 + skip_chans
            n_chans = n_chans if not final_div else skip_chans
        self.decoder = nn.Sequential(*decoder, PixelShuffleICNR(n_chans, n_chans))
        self.head = nn.Sequential(
            nn.Conv2d(n_chans + input_shape[0], n_chans, 1),
            LastCross(n_chans, norm_layer=norm_layer),
            nn.Conv2d(n_chans, n_classes, 1),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = interpolate(y, x.shape[-2:], mode="nearest")
        y = torch.cat([x, y], dim=1)
        y = self.head(y)
        return y

    def _register_output_hooks(self, input_shape=(3, 224, 224)):
        sizes, modules = get_sizes(self.encoder, input_shape=input_shape)
        mods = []
        idxs = np.where(sizes[:-1, -1] != sizes[1:, -1])[0]

        def _hook(model, input, output):
            return output

        for k in idxs[::-1]:
            m = modules[k]
            if "downsample" not in m.name:
                mods.append(m)
        self.hooks = Hooks(mods, _hook)

        return sizes, idxs

    def __del__(self):
        if hasattr(self, "hooks"):
            self.hooks.remove()


class Normalizer(nn.Module):
    """
    Normalizer module that takes a RGB image as input and returns the normalized version
    using HEnorm method.

    Args:
        encoder_name: name of the encoder to use. Can be any model registered in timm.
        input_size: side size the input image.
        pretrained: whether to use a pretrained model. Only available for timm encoders.
    """

    def __init__(self, encoder_name, input_size=1024, pretrained=True):
        super().__init__()
        input_shape = (3, input_size, input_size)
        self.unet = DynamicUnet(
            encoder_name,
            n_classes=3,
            input_shape=input_shape,
            pretrained=pretrained,
        )
        self.color_converter = ColorConverter("h")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.color_converter(x).detach()
        return self.unet(x)

    def freeze_encoder(self):
        """
        Freeze the encoder part of the normalizer.
        """
        for m in named_leaf_modules(self):
            if "encoder" in m.name and not isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = False

    def init_bn(self):
        """
        Initialize BatchNorm layers with bias `1e-3` and weights `1`.
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                with torch.no_grad():
                    m.bias.fill_(1e-3)
                    m.weight.fill_(1.0)

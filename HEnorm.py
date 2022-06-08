import torch.nn as nn
import torch
import timm
import numpy as np
from torch.nn.functional import interpolate

from color_converter import ColorConverter
from modules import (
    ConvBnRelu,
    DecoderBlock,
    Hooks,
    LastCross,
    PixelShuffleICNR,
    get_sizes,
    named_leaf_modules,
)


class DynamicUnet(nn.Module):
    """"""

    def __init__(
        self, encoder_name, n_classes=2, input_shape=(3, 224, 224), pretrained=True
    ):
        super().__init__()
        encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            pretrained_strict=False,
        )
        cut = -2

        self.encoder = nn.Sequential(*(list(encoder.children())[:cut] + [nn.ReLU()]))
        encoder_sizes, idxs = self._register_output_hooks(input_shape=input_shape)
        n_chans = int(encoder_sizes[-1][1])
        middle_conv = nn.Sequential(
            ConvBnRelu(n_chans, n_chans // 2, 3),
            ConvBnRelu(n_chans // 2, n_chans, 3),
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
                )
            )
            n_chans = n_chans // 2 + skip_chans
            n_chans = n_chans if not final_div else skip_chans
        self.decoder = nn.Sequential(*decoder, PixelShuffleICNR(n_chans, n_chans))
        self.head = nn.Sequential(
            nn.Conv2d(n_chans + input_shape[0], n_chans, 1),
            LastCross(n_chans),
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
    """"""

    def __init__(self, model_name, input_size=1024, pretrained=True):
        super().__init__()
        input_shape = (3, input_size, input_size)
        self.unet = DynamicUnet(
            model_name,
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

import numpy as np
import torch
import torch.nn as nn

_rgb_from_hed = torch.tensor(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=torch.float32
)


def rgb_to_hed(image: torch.Tensor, conversion_matrix: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4:
        perm1 = (0, 2, 3, 1)
        perm2 = (0, 3, 1, 2)
    else:
        perm1 = (1, 2, 0)
        perm2 = (2, 0, 1)
    image += 2
    stains = -(torch.log(image) / np.log(10)).permute(*perm1) @ conversion_matrix
    return stains.permute(*perm2)


def rgb_to_h(image: torch.Tensor, conversion_matrix: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4:
        h = rgb_to_hed(image, conversion_matrix)[:, 0]
        h = (h + 0.7) / 0.46
        return torch.stack((h, h, h), axis=1)
    else:
        h = rgb_to_hed(image, conversion_matrix)[0]
        h = (h + 0.7) / 0.46
        return torch.stack((h, h, h), axis=0)


def rgb_to_e(image: torch.Tensor, conversion_matrix: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4:
        e = rgb_to_hed(image, conversion_matrix)[:, 1]
        e = (e + 0.1) / 0.47
        return torch.stack((e, e, e), axis=1)
    else:
        e = rgb_to_hed(image, conversion_matrix)[1]
        e = (e + 0.1) / 0.47
        return torch.stack((e, e, e), axis=0)


class ColorConverter(nn.Module):
    def __init__(self, open_mode="rgb"):
        super().__init__()
        self.open_mode = open_mode.lower()
        if self.open_mode in ("h", "e", "hed"):
            self.conversion_matrix = nn.Parameter(
                torch.inverse(_rgb_from_hed), requires_grad=False
            )
        else:
            self.conversion_matrix = None

    def forward(self, x):
        if self.open_mode != "rgb":
            open_func = globals()[f"rgb_to_{self.open_mode}"]
            if self.conversion_matrix is None:
                x = open_func(x)
            else:
                x = open_func(x, self.conversion_matrix)
        return x

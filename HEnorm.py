import torch.nn as nn
import torch
import timm
import numpy as np
from torch.nn.functional import interpolate

from color_converter import ColorConverter


class Hook:
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (
                (o.detach() for o in input)
                if isinstance(input, (tuple, list))
                else input.detach()
            )
            output = (
                (o.detach() for o in output)
                if isinstance(output, (tuple, list))
                else output.detach()
            )
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks:
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True):
        for k, m in enumerate(ms):
            setattr(self, f"hook_{k}", Hook(m, hook_func, is_forward, detach))
        # self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]
        self.n = len(ms)

    def __getitem__(self, i):
        return getattr(self, f"hook_{i}")

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter([self[k] for k in range(len(self))])

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def named_leaf_modules(model, name=""):
    named_children = list(model.named_children())
    if named_children == []:
        model.name = name
        return [model]
    else:
        res = []
        for n, m in named_children:
            if not isinstance(m, torch.jit.ScriptModule):
                pref = name + "." if name != "" else ""
                res += named_leaf_modules(m, pref + n)
        return res


def get_sizes(model, input_shape=(3, 224, 224), leaf_modules=None):
    leaf_modules = (
        leaf_modules if leaf_modules is not None else named_leaf_modules(model)
    )

    class Count:
        def __init__(self):
            self.k = 0

    count = Count()

    def _hook(model, input, output):
        model.k = count.k
        count.k += 1
        return model, output.shape

    with Hooks(leaf_modules, _hook) as hooks:
        x = torch.rand(1, *input_shape)
        model.cpu().eval()(x)
        sizes = [list(hook.stored[1]) for hook in hooks if hook.stored is not None]
        mods = [hook.stored[0] for hook in hooks if hook.stored is not None]
    idxs = np.argsort([mod.k for mod in mods])
    return np.array(sizes, dtype=object)[idxs], np.array(mods, dtype=object)[idxs]


def bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None, norm_layer=nn.BatchNorm1d):
    layers = [norm_layer(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        eps=1e-5,
        norm_layer=nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        self.bn = norm_layer(out_channels, eps=eps)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBn(nn.Module):
    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        eps=1e-5,
        momentum=0.01,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_chans, out_channels, bias=True, scale_factor=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_channels * scale_factor ** 2, 1, bias=bias, **kwargs
        )
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale_factor)
        # self.pad = nn.ReflectionPad2d((1, 0, 1, 0))
        # self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.shuf(x)
        # x = self.pad(x)
        # x = self.blur(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        skip_chans,
        hook,
        final_div=True,
        norm_layer=nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffleICNR(in_chans, in_chans // 2, **kwargs)
        self.bn = norm_layer(skip_chans)
        ni = in_chans // 2 + skip_chans
        nf = ni if not final_div else skip_chans
        self.relu = nn.ReLU()
        self.conv1 = ConvBnRelu(ni, nf, 3, padding=1, norm_layer=norm_layer, **kwargs)
        self.conv2 = ConvBnRelu(nf, nf, 3, padding=1, norm_layer=norm_layer, **kwargs)

    def forward(self, x):
        skipco = self.hook.stored
        x = self.shuf(x)
        ssh = skipco.shape[-2:]
        if ssh != x.shape[-2:]:
            x = interpolate(x, ssh, mode="nearest")
        x = self.relu(torch.cat([x, self.bn(skipco)], dim=1))
        return self.conv2(self.conv1(x))


class LastCross(nn.Module):
    def __init__(self, n_chans, bottle=False, norm_layer=nn.BatchNorm2d):
        super(LastCross, self).__init__()
        n_mid = n_chans // 2 if bottle else n_chans
        self.conv1 = ConvBnRelu(n_chans, n_mid, 3, padding=1, norm_layer=norm_layer)
        self.conv2 = ConvBnRelu(n_mid, n_chans, 3, padding=1, norm_layer=norm_layer)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


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

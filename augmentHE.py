from numbers import Number
from typing import Dict, Union, Any, List, Optional

import numpy as np
from albumentations import ImageOnlyTransform
from nptyping import NDArray
from staintools.miscellaneous.get_concentrations import get_concentrations
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor


class StainAugmentor(ImageOnlyTransform):
    """
    Albumentation transform class implementation of AugmentHE.

    Args:
        alpha_range: defines the range to use when randomly picking the multiplicative
            coefficients for the density matrix. The draw interval is defined as
            [1-alpha_range, 1+alpha_range].
        beta_range: defines the range to use when randomly picking the additive
            coefficients for the density matrix. The draw interval is defined as
            [-beta_range, beta_range].
        alpha_stain_range: defines the range to use when randomly picking the
            multiplicative coefficients for the stain matrix. The draw interval is
            defined as [1-alpha_stain_range, 1+alpha_stain_range].
        beta_stain_range: defines the range to use when randomly picking the
            additive coefficients for the stain matrix. The draw interval is
            defined as [-beta_stain_range, beta_stain_range].
        he_ratio: ratio between the H coefficients and the E coefficients. Rangeq
            defined above are multiplied by this value when drawing H coefficients.
        always_apply: whether to always apply this transform.
        p: probability to apply this transform.
    """

    def __init__(
        self,
        alpha_range: float = 0.4,
        beta_range: float = 0.4,
        alpha_stain_range: float = 0.3,
        beta_stain_range: float = 0.2,
        he_ratio: float = 0.3,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(StainAugmentor, self).__init__(always_apply, p)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.alpha_stain_range = alpha_stain_range
        self.beta_stain_range = beta_stain_range
        self.he_ratio = he_ratio

    def get_params(
        self,
    ) -> Dict[str, Union[NDArray[(2,), float], NDArray[(2, 3), float]]]:
        return {
            "alpha": np.random.uniform(
                1 - self.alpha_range, 1 + self.alpha_range, size=2
            ),
            "beta": np.random.uniform(-self.beta_range, self.beta_range, size=2),
            "alpha_stain": np.stack(
                (
                    np.random.uniform(
                        1 - self.alpha_stain_range * self.he_ratio,
                        1 + self.alpha_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        1 - self.alpha_stain_range,
                        1 + self.alpha_stain_range,
                        size=3,
                    ),
                ),
            ),
            "beta_stain": np.stack(
                (
                    np.random.uniform(
                        -self.beta_stain_range * self.he_ratio,
                        self.beta_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        -self.beta_stain_range, self.beta_stain_range, size=3
                    ),
                ),
            ),
        }

    def initialize(self, alpha, beta, shape=2):
        alpha = alpha if alpha is not None else np.ones(shape)
        beta = beta if beta is not None else np.zeros(shape)
        return alpha, beta

    def apply(
        self,
        image: NDArray[(Any, Any, 3), Number],
        alpha: Optional[NDArray[(2,), float]] = None,
        beta: Optional[NDArray[(2,), float]] = None,
        alpha_stain: Optional[NDArray[(2, 3), float]] = None,
        beta_stain: Optional[NDArray[(2, 3), float]] = None,
        **params
    ) -> NDArray[(Any, Any, 3), Number]:
        alpha, beta = self.initialize(alpha, beta, shape=2)
        alpha_stain, beta_stain = self.initialize(alpha_stain, beta_stain, shape=(2, 3))
        if not image.dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        stain_matrix = VahadaneStainExtractor.get_stain_matrix(image)
        HE = get_concentrations(image, stain_matrix)
        stain_matrix = stain_matrix * alpha_stain + beta_stain
        stain_matrix = np.clip(stain_matrix, 0, 1)
        HE = np.where(HE > 0.2, HE * alpha[None] + beta[None], HE)
        out = np.exp(-np.dot(HE, stain_matrix)).reshape(image.shape)
        out = np.clip(out, 0, 1)
        return out.astype(np.float32)

    def get_transform_init_args_names(self) -> List:
        return (
            "alpha_range",
            "beta_range",
            "alpha_stain_range",
            "beta_stain_range",
            "he_ratio",
        )

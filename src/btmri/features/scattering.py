"""2D wavelet scattering descriptors (Bruna & Mallat 2013; Kymatio, Andreux et al. 2020).

Scattering is the principled, stable successor of the Gabor + DWT descriptors
of the original repository: cascades of Morlet wavelet modulus operators with
local averaging, translation-invariant up to 2^J pixels and Lipschitz-stable to
small deformations. Slices are downsampled to 128 x 128 (1.875 mm pixels);
with J = 3 and L = 8 orientations there are 1 + JL + L^2 J(J-1)/2 = 217
channels, log-transformed and averaged over 2 x 2 spatial quadrants (868 dims).
"""

from __future__ import annotations

import numpy as np
from skimage.transform import resize

SIZE = 128
J = 3
L = 8


def _downsample(images: np.ndarray) -> np.ndarray:
    return np.stack([resize(im.astype(np.float32) / 255.0, (SIZE, SIZE), order=1, anti_aliasing=True)
                     for im in images]).astype(np.float32)


def batch_scattering(images: np.ndarray, batch: int = 64) -> np.ndarray:
    # import the 2D frontend directly: kymatio.numpy also imports the 3D module,
    # which depends on scipy.special.sph_harm (removed in recent SciPy releases)
    from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D

    S = ScatteringNumPy2D(J=J, shape=(SIZE, SIZE), L=L)
    out = []
    for i in range(0, len(images), batch):
        x = _downsample(images[i : i + batch])
        s = S(x)  # (b, 217, 16, 16)
        s = np.log(np.abs(s) + 1e-4)
        b, c, h, w = s.shape
        q = s.reshape(b, c, 2, h // 2, 2, w // 2).mean(axis=(3, 5))  # 2x2 quadrants
        out.append(q.reshape(b, -1).astype(np.float32))
    return np.concatenate(out)

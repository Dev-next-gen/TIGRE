"""CTnoise.add: the default photon count, and readings that fall to zero or below.

The reference is MATLAB/Utilities/addCTnoise.m: I0 defaults to 60000 (or
max(proj)/5 when the projections exceed that), and after the noise is added
every reading <= 0 is set to 1e-6 before the log.
"""
import numpy as np

from tigre.utilities import CTnoise


def _projections():
    # line integrals shaped like a real sinogram: 0 at the edges, a few hundred in the middle
    u = np.linspace(-1, 1, 64, dtype=np.float32)
    row = 300 * np.sqrt(np.clip(1 - u**2, 0, None))
    return np.tile(row, (16, 32, 1)).astype(np.float32)


def test_default_poisson_gives_finite_low_noise_projections():
    proj = _projections()
    noisy = CTnoise.add(proj)
    assert np.isfinite(noisy).all()
    # 60000 photons: the noise is a small fraction of the signal
    assert np.abs(noisy - proj).mean() < 0.05 * proj.max()


def test_low_photon_count_does_not_produce_nan_or_inf():
    proj = _projections()
    noisy = CTnoise.add(proj, Poisson=5, Gaussian=np.array([0, 2]))
    assert np.isfinite(noisy).all()

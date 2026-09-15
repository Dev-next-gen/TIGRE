"""MSSIM in Measure_Quality must agree with MATLAB/Utilities/Quality_measures/MSSIM.m."""
import numpy as np
import pytest

from tigre.utilities.Measure_Quality import Measure_Quality


def mssim_matlab(real, res):
    """Literal transcription of MATLAB/Utilities/Quality_measures/MSSIM.m."""
    real = real.ravel().astype(np.float64)
    res = res.ravel().astype(np.float64)
    N = real.size

    meanreal = real.mean()
    meanres = res.mean()

    K1 = 0.01
    d = real.max() - real.min()
    l = ((2 * meanreal * meanres) + (K1 * d) ** 2) / (
        meanreal ** 2 + meanres ** 2 + (K1 * d) ** 2
    )

    K2 = 0.03
    sreal = real.std(ddof=1)
    sres = res.std(ddof=1)
    c = ((2 * sreal * sres) + (K2 * d) ** 2) / (sreal ** 2 + sres ** 2 + (K2 * d) ** 2)

    delta = np.sum((real - meanreal) * (res - meanres)) / (N - 1)
    s = (delta + ((K2 * d) ** 2) / 2) / ((sreal * sres) + ((K2 * d) ** 2) / 2)

    return (1 / N) * l * c * s


class TestMSSIM:
    def test_identical_images_give_unit_similarity(self):
        """l, c and s are each exactly 1 when both images are the same, so
        MSSIM is 1/N whatever the dynamic range of the data."""
        rng = np.random.default_rng(0)
        img = rng.random((8, 8, 8)) * 255.0  # dynamic range far from 1

        got = float(Measure_Quality(img, img, ["MSSIM"]))

        assert got == pytest.approx(1.0 / img.size, rel=1e-12)

    def test_matches_matlab_reference(self):
        rng = np.random.default_rng(1)
        real = rng.random((8, 8, 8)) * 255.0
        res = real + rng.normal(scale=10.0, size=real.shape)

        got = float(Measure_Quality(real, res, ["MSSIM"]))

        assert got == pytest.approx(mssim_matlab(real, res), rel=1e-12)

    def test_matches_matlab_reference_unit_dynamic_range(self):
        """Same comparison as above, on data with a unit dynamic range (d = 1)."""
        rng = np.random.default_rng(2)
        real = rng.random((6, 6, 6))
        real[0, 0, 0], real[0, 0, 1] = 0.0, 1.0
        res = real + rng.normal(scale=0.05, size=real.shape)

        got = float(Measure_Quality(real, res, ["MSSIM"]))

        assert got == pytest.approx(mssim_matlab(real, res), rel=1e-12)

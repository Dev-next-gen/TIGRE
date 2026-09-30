# -*- coding: utf-8 -*-
"""Wang data-redundancy weights for an offset (half-fan) detector.

Python port of MATLAB/Utilities/redundancy_weighting.m and
MATLAB/Utilities/apply_wang_weights.m.

Reference: Ge Wang, "X-ray micro-CT with a displaced detector array",
Medical Physics 29(7):1634-1636, 2002. DOI: 10.1118/1.1489043
Use in iterative reconstruction: Rit et al., Physics in Medicine & Biology
66(17):175015, 2021. DOI: 10.1088/1361-6560/ac16bc

When the detector is shifted along its fan direction (`offDetector[1]`), a full
turn oversamples the region that stays inside both the direct and the conjugate
view of the detector, and samples the rest once. Every ray in the twice-covered
region therefore enters the data term twice, and the ray at the edge of that
region enters it once, so the sampling density jumps by a factor of two at the
transition radius. That step is what shows up as a ring at a fixed radius.

The Wang weight w is a smooth ramp across the twice-covered region built so
that a ray and its conjugate always add up to the same total as a ray that is
only seen once:

    w(t) + w(-t) = 2   for |t| <= abstheta   (both views contribute)
    w(t)         = 2   for  t  >  abstheta   (only this view contributes)
    w(t)         = 0   for  t  <  -abstheta  (masked, conjugate carries it)

so the accumulated weight is 2 everywhere and the step disappears.

Index convention: Python detector arrays are [V, U] and MATLAB's are [U, V],
so every MATLAB `geo.*Detector(1)` (the fan direction) is `geo.*Detector[1]`
here and `geo.*Detector(2)` is `geo.*Detector[0]`. The returned matrix is
(nDetector[0], nDetector[1]), which is MATLAB's [nDetector(2), nDetector(1)]:
the same (V, U) shape, broadcastable against a projection stack shaped
(angles, V, U).

--------------------------------------------------------------------
This file is part of the TIGRE Toolbox

Copyright (c) 2015, University of Bath and
                    CERN-European Organization for Nuclear Research
                    All rights reserved.

License:            Open Source under BSD.
                    See the full license at
                    https://github.com/CERN/TIGRE/license.txt

Contact:            tigre.toolbox@gmail.com
Codes:              https://github.com/CERN/TIGRE/
--------------------------------------------------------------------
"""
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np

# Ge Wang.
# X-ray micro-CT with a displaced detector array.
# Medical Physics, 29(7):1634-1636, 2002.
#
# @article{wang2002displaced,
#   title={X-ray micro-CT with a displaced detector array},
#   author={Wang, Ge},
#   journal={Medical Physics},
#   volume={29},
#   number={7},
#   pages={1634--1636},
#   year={2002},
#   publisher={American Association of Physicists in Medicine}
# }

# Percentage offset above which MATLAB warns that artifacts are likely.
MAX_SAFE_OFFSET_PERCENT = 30


def _fan_offsets(geo):
    """Detector offsets along the fan direction, as a 1-D array, one per angle.

    `check_geo` expands `offDetector` to (nangles, 2); a geometry that has not
    been checked yet still carries the plain (2,) form. The fan direction is
    the last axis in both.
    """
    off = np.asarray(geo.offDetector)
    if off.ndim == 1:
        return off[1:2]
    return off[:, 1]


def _first(value):
    """First element of a geometry entry that `check_geo` may have expanded."""
    return float(np.asarray(value).ravel()[0])


def apply_wang_weights(geo):
    """Whether Wang weighting applies to this geometry.

    Port of MATLAB/Utilities/apply_wang_weights.m. False for a centred
    detector, and False with a warning when the geometry varies per angle, in
    which case a single weight matrix cannot describe the scan.
    """
    offsets = _fan_offsets(geo)
    if offsets.size > 1 and np.unique(offsets).size > 1:
        warnings.warn("Wang weights: varying offDetector detected, Wang weights not being applied")
        return False

    if offsets[0] == 0:
        return False

    dso = np.asarray(geo.DSO).ravel()
    if dso.size > 1 and np.unique(dso).size > 1:
        warnings.warn("Wang weights: varying DSO detected, Wang weights not being applied")
        return False

    percent_offset = abs(offsets[0] / _first(geo.sDetector[1])) * 100
    if percent_offset > MAX_SAFE_OFFSET_PERCENT:
        warnings.warn(
            "Wang weights: Detector offset percent: %0.2f is greater than %d which may result "
            "in image artifacts, consider rebinning 360 degree projections to 180 degrees"
            % (percent_offset, MAX_SAFE_OFFSET_PERCENT)
        )

    return True


def redundancy_weighting(geo):
    """Wang data-redundancy weights for `geo`, shaped (nDetector[0], nDetector[1]).

    Returns all ones when the geometry does not call for weighting, so the
    result can always be multiplied in unconditionally.

    :param geo: tigre.utilities.geometry.Geometry
    :return: np.array(dtype=float32), shape (nDetector[0], nDetector[1])
    """
    n_v = int(geo.nDetector[0])
    n_u = int(geo.nDetector[1])
    w = np.ones((n_v, n_u), dtype=np.float32)

    if not apply_wang_weights(geo):
        return w

    dsd = _first(geo.DSD)
    dso = _first(geo.DSO)
    cor = _first(geo.COR) if hasattr(geo, "COR") else 0.0
    off_detector = _fan_offsets(geo)[0]

    # Detector coordinates referred to the rotation axis, as MATLAB's `us`.
    offset = off_detector + (dsd / dso) * cor
    us = (np.arange(n_u) - n_u / 2 + 0.5) * _first(geo.dDetector[1]) + abs(offset)
    us = us * dso / dsd

    # Half-width of the twice-covered region, referred to the rotation axis.
    theta = (_first(geo.sDetector[1]) / 2 - abs(off_detector)) * np.sign(off_detector)
    abstheta = abs(theta * dso / dsd)

    ramp = np.ones(n_u)
    inside = np.abs(us) <= abstheta
    if abstheta > 0:
        ramp[inside] = 0.5 * (
            np.sin((np.pi / 2) * np.arctan(us[inside] / dso) / np.arctan(abstheta / dso)) + 1
        )
    ramp[us < -abstheta] = 0
    ramp = ramp * 2

    w *= ramp.astype(np.float32)
    if theta < 0:
        w = np.fliplr(w)

    return np.ascontiguousarray(w, dtype=np.float32)

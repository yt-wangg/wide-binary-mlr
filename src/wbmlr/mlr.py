"""Dynamical mass--luminosity relation of single main-sequence stars from Gaia wide binaries.

The relation is a monotone tensor-product cubic B-spline surface ``log10(M/Msun) = g(M_G, Z)`` with 8 x 4
coefficients, fitted to the relative motion of 14,876 wide binaries (Wang et al., in prep.).  ``Z`` is the
calibrated metallicity on the PARSEC [M/H] scale, not the raw J-CaPS value (see :func:`calibrate_metallicity`).
The posterior is stored as 4000 coefficient draws, so uncertainties can be propagated and are correlated
between magnitudes and metallicities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, RegularGridInterpolator

DATA = Path(__file__).parent / "data"
MG_RANGE = (3.5, 13.5)
Z_RANGE = (-1.0, 0.6)
MG_SUN = 4.66  # magnitude at which the J-CaPS bias is fixed to zero


def _basis(x, knots, degree):
    """B-spline basis values, shape ``x.shape + (n_coefficients,)``."""
    x = np.asarray(x, dtype=float)
    n = len(knots) - degree - 1
    return BSpline(knots, np.eye(n), degree, extrapolate=False)(np.clip(x, knots[0], knots[-1]))


def calibrate_metallicity(mh_jcaps, mg):
    """Calibrated metallicity ``Z = [M/H]_JC - b_G(M_G)`` of a Gaia XP (J-CaPS) metallicity.

    ``b_G`` is the relative bias of J-CaPS, a piecewise-linear function of the absolute magnitude, referenced to
    zero at the solar magnitude (the absolute zero point is anchored to the Sun).
    """
    t = np.loadtxt(DATA / "jcaps_bias.csv", delimiter=",", skiprows=1)
    bias = np.interp(mg, t[:, 0], t[:, 1]) - np.interp(MG_SUN, t[:, 0], t[:, 1])
    return np.asarray(mh_jcaps, dtype=float) - bias


class MLR:
    """Posterior of the mass--luminosity relation.

    Examples
    --------
    >>> mlr = MLR()
    >>> mlr.mass(5.0, 0.0)                     # median mass in Msun
    >>> lo, med, hi = mlr.mass_quantiles(5.0, 0.0)      # 16, 50, 84 per cent
    >>> draws = mlr.mass_draws(mg, z, n=200)   # (200, ...) posterior draws, for error propagation
    """

    def __init__(self):
        d = np.load(DATA / "mlr_theta_draws.npz")
        self.theta = d["theta"]                    # (n_draws, 8, 4)
        self.knots_x, self.knots_z = d["knots_x"], d["knots_z"]
        self.degree_x, self.degree_z = int(d["degree_x"]), int(d["degree_z"])
        c = np.load(DATA / "data_coverage.npz")
        self._cover = RegularGridInterpolator((c["z"], c["mg"]), c["density"], bounds_error=False, fill_value=0.0)
        self._level = float(c["level"])

    @property
    def n_draws(self) -> int:
        return self.theta.shape[0]

    def _prepare(self, mg, z):
        mg, z = np.broadcast_arrays(np.asarray(mg, dtype=float), np.asarray(z, dtype=float))
        out_of_range = (mg < MG_RANGE[0]) | (mg > MG_RANGE[1]) | (z < Z_RANGE[0]) | (z > Z_RANGE[1])
        return mg, z, out_of_range

    def log_mass_draws(self, mg, z, n=None, seed=0):
        """Posterior draws of ``log10(M/Msun)``, shape ``(n_draws,) + broadcast(mg, z).shape``.

        Outside the fitted domain (3.5 <= M_G <= 13.5, -1.0 <= Z <= 0.6) the values are NaN.
        """
        mg, z, bad = self._prepare(mg, z)
        theta = self.theta
        if n is not None and n < self.n_draws:
            theta = theta[np.random.default_rng(seed).choice(self.n_draws, size=int(n), replace=False)]
        bx = _basis(mg, self.knots_x, self.degree_x)
        bz = _basis(z, self.knots_z, self.degree_z)
        g = np.einsum("...i,...h,dih->d...", bx, bz, theta)
        g[:, bad] = np.nan
        return g

    def mass_draws(self, mg, z, n=None, seed=0):
        """Posterior draws of the mass in solar masses (see :meth:`log_mass_draws`)."""
        return 10.0 ** self.log_mass_draws(mg, z, n=n, seed=seed)

    def mass_quantiles(self, mg, z, q=(16, 50, 84)):
        """Quantiles (per cent) of the mass, shape ``(len(q),) + broadcast(mg, z).shape``."""
        return np.percentile(self.mass_draws(mg, z), q, axis=0)

    def mass(self, mg, z):
        """Posterior median mass in solar masses."""
        return self.mass_quantiles(mg, z, q=(50,))[0]

    def in_data_region(self, mg, z):
        """True where the fit is constrained by the data.

        The region encloses 95 per cent of the sample components in the (M_G, Z) plane.  Outside it the relation is
        determined by the monotonicity and the priors, and should not be treated as a measurement.
        """
        mg, z, bad = self._prepare(mg, z)
        pts = np.stack([z, mg], axis=-1).reshape(-1, 2)
        return (self._cover(pts) >= self._level).reshape(mg.shape) & ~bad

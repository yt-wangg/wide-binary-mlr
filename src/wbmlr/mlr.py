"""Dynamical mass--luminosity relation of single main-sequence stars from Gaia wide binaries.

The relation is a monotone tensor-product cubic B-spline surface ``log10(M/Msun) = g(M_G, Z)`` with 8 x 4
coefficients, fitted to the relative motion of 14,876 wide binaries (Wang et al., in prep.).  ``Z`` is the
metallicity on the PARSEC [M/H] scale with the zero point anchored to the Sun.
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


def _basis(x, knots, degree):
    """B-spline basis values, shape ``x.shape + (n_coefficients,)``."""
    x = np.asarray(x, dtype=float)
    n = len(knots) - degree - 1
    return BSpline(knots, np.eye(n), degree, extrapolate=False)(np.clip(x, knots[0], knots[-1]))


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
        g = np.load(DATA / "parsec_color_grid.npz")
        self._cmg, self._cz, self._ccolor = g["mg"], g["z"], g["color"]   # PARSEC colour (n_mg, n_z)

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

    def z_from_color(self, mg, bp_rp):
        """Metallicity ``Z`` implied by a colour--magnitude position through the PARSEC main sequence.

        ``bp_rp`` must be the intrinsic (dereddened) ``BP-RP``. At fixed ``M_G`` the PARSEC colour increases
        monotonically with metallicity (logAge = 9.6), and the relation is inverted over ``-1.0 <= Z <= 0.6``.
        Positions bluer or redder than the PARSEC colours at the edges of that range return NaN.
        Compared with the spectroscopic metallicities of the calibration sample the colour-based ``Z`` has a
        scatter of about 0.15 dex, which is propagated by ``mass_from_cmd`` (``z_sigma``).
        """
        mg, c = np.broadcast_arrays(np.asarray(mg, dtype=float), np.asarray(bp_rp, dtype=float))
        out = np.full(mg.shape, np.nan)
        ok = (mg >= MG_RANGE[0]) & (mg <= MG_RANGE[1])
        idx = np.flatnonzero(ok.ravel())
        mgf, cf = mg.ravel()[idx], c.ravel()[idx]
        # colour of each Z node at the requested magnitudes (linear in M_G on the 0.01 mag grid)
        cols = np.stack([np.interp(mgf, self._cmg, self._ccolor[:, j]) for j in range(self._cz.size)], axis=1)
        z = np.array([np.interp(cf[i], cols[i], self._cz, left=np.nan, right=np.nan) for i in range(idx.size)])
        res = out.ravel(); res[idx] = z
        return res.reshape(mg.shape)

    def mass_from_cmd(self, mg, bp_rp, z_sigma=0.0, n=None, seed=0):
        """Mass quantiles (16, 50, 84 per cent) of a star placed on the colour--magnitude diagram.

        The metallicity is taken from the colour (:meth:`z_from_color`). With ``z_sigma > 0`` a Gaussian
        uncertainty of that size (dex) in ``Z`` is added to the posterior draws (0.15 is typical for the colour-based
        metallicity), so the quantiles include the metallicity error. Returns an array ``(3,) + shape``.
        """
        z = self.z_from_color(mg, bp_rp)
        if z_sigma and z_sigma > 0:
            rng = np.random.default_rng(seed)
            nd = self.n_draws if n is None else min(int(n), self.n_draws)
            zz = np.clip(z[None] + z_sigma * rng.standard_normal((nd,) + z.shape), Z_RANGE[0], Z_RANGE[1])
            sel = np.random.default_rng(seed + 1).choice(self.n_draws, size=nd, replace=False)
            bx = _basis(np.broadcast_to(np.asarray(mg, dtype=float), z.shape), self.knots_x, self.degree_x)
            bz = _basis(zz, self.knots_z, self.degree_z)
            g = np.einsum("...i,d...h,dih->d...", bx, bz, self.theta[sel])
            return np.percentile(10.0 ** g, [16, 50, 84], axis=0)
        return self.mass_quantiles(mg, z)

    def in_data_region(self, mg, z):
        """True where the fit is constrained by the data.

        The region encloses 95 per cent of the sample components in the (M_G, Z) plane.  Outside it the relation is
        determined by the monotonicity and the priors, and should not be treated as a measurement.
        """
        mg, z, bad = self._prepare(mg, z)
        pts = np.stack([z, mg], axis=-1).reshape(-1, 2)
        return (self._cover(pts) >= self._level).reshape(mg.shape) & ~bad

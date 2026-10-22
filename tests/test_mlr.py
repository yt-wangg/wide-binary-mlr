import csv
from pathlib import Path

import numpy as np
import pytest

from wbmlr import MLR
from wbmlr.mlr import DATA


@pytest.fixture(scope="module")
def mlr():
    return MLR()


def test_grid_table_is_reproduced(mlr):
    rows = list(csv.DictReader(open(DATA / "mlr_grid.csv")))
    pick = rows[::397]
    mg = np.array([float(r["MG"]) for r in pick]); z = np.array([float(r["Z"]) for r in pick])
    p16, p50, p84 = mlr.mass_quantiles(mg, z)
    assert np.allclose(p50, [float(r["mass_p50"]) for r in pick], rtol=1e-5)
    assert np.allclose(p16, [float(r["mass_p16"]) for r in pick], rtol=1e-5)
    assert np.allclose(p84, [float(r["mass_p84"]) for r in pick], rtol=1e-5)


def test_monotone_in_magnitude_and_metallicity(mlr):
    mg = np.linspace(3.5, 13.5, 101)[:, None]; z = np.linspace(-1, 0.6, 33)[None, :]
    g = mlr.log_mass_draws(mg, z, n=200)
    assert np.all(np.diff(g, axis=1) <= 1e-9)    # fainter -> less massive
    assert np.all(np.diff(g, axis=2) >= -1e-9)   # more metal-rich -> more massive


def test_solar_anchor(mlr):
    assert abs(mlr.mass(4.67, 0.0) - 1.0) < 0.02


def test_shapes_and_broadcasting(mlr):
    assert mlr.mass(5.0, 0.0).shape == ()
    assert mlr.mass(np.array([5.0, 6.0]), 0.0).shape == (2,)
    assert mlr.mass_draws(np.ones((3, 4)) * 5, 0.0, n=10).shape == (10, 3, 4)
    lo, med, hi = mlr.mass_quantiles(7.0, 0.1)
    assert lo < med < hi


def test_outside_domain_is_nan(mlr):
    assert np.isnan(mlr.mass(3.0, 0.0)) and np.isnan(mlr.mass(5.0, 1.0))


def test_data_region(mlr):
    assert bool(mlr.in_data_region(5.0, 0.0))
    assert not bool(mlr.in_data_region(13.0, -0.9))
    assert not bool(mlr.in_data_region(2.0, 0.0))


def test_cmd_table_is_reproduced(mlr):
    rows = list(csv.DictReader(open(DATA / "mlr_cmd_grid.csv")))[::401]
    mg = np.array([float(r["MG"]) for r in rows]); c = np.array([float(r["BP_RP"]) for r in rows])
    z, zs = mlr.z_from_color(mg, c, return_std=True)
    assert np.allclose(z, [float(r["Z_mean"]) for r in rows], atol=2e-3)
    assert np.allclose(zs, [float(r["Z_std"]) for r in rows], atol=2e-3)
    q = mlr.mass_from_cmd(mg, c, include_z_scatter=False)
    assert np.allclose(q[1], [float(r["mass_p50"]) for r in rows], rtol=2e-3)


def test_cmd_support(mlr):
    assert np.isfinite(mlr.z_from_color(8.0, 1.75))
    assert np.isnan(mlr.z_from_color(8.0, 0.3)) and np.isnan(mlr.z_from_color(2.0, 1.0))
    assert np.isnan(mlr.mass_from_cmd(8.0, 0.3)).all()


def test_z_scatter_widens_the_interval(mlr):
    a = mlr.mass_from_cmd(8.0, 1.75, include_z_scatter=False)
    b = mlr.mass_from_cmd(8.0, 1.75)
    assert (b[2] - b[0]) > (a[2] - a[0])
    assert abs(b[1] / a[1] - 1) < 0.03


def test_colour_table_of_the_grid():
    rows = [r for r in csv.DictReader(open(DATA / "mlr_grid.csv")) if r["bp_rp_mean"] != "nan"]
    z = {}
    for r in rows:
        if abs(float(r["MG"]) - 8.0) < 1e-9:
            z[float(r["Z"])] = float(r["bp_rp_mean"])
    zs = sorted(z)
    assert len(zs) > 5 and all(z[b] > z[a] for a, b in zip(zs[:-1], zs[1:]))   # redder when more metal-rich

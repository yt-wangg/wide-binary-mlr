import csv
from pathlib import Path

import numpy as np
import pytest

from wbmlr import MLR, calibrate_metallicity
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


def test_calibrate_metallicity():
    assert np.isclose(calibrate_metallicity(0.1, 4.66), 0.1)           # zero bias at the solar magnitude
    assert calibrate_metallicity(0.0, 11.0) > 0.0                      # b_G < 0 for faint stars

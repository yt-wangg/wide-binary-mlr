# wbmlr: a dynamical mass–luminosity relation from Gaia wide binaries

Mass of a single main-sequence star as a function of its Gaia absolute magnitude $M_G$ and metallicity $Z$,
calibrated dynamically with 14,876 wide binaries (Wang, Rix, Li & Ting, in prep.). The mass scale comes from the
relative motion of the two stars, not from stellar evolution models.

- Domain: $3.5 \le M_G \le 13.5$ mag, $-1.0 \le Z \le 0.6$ dex (NaN outside).
- Data region: $3.6 \lesssim M_G \lesssim 12.1$, $-0.5 \lesssim Z \lesssim 0.45$. Outside it the relation is set by
  monotonicity and priors, not by the data; `in_data_region` flags it.
- Statistical uncertainty where the data are: 0.8–3.7 % in mass (median 1.9 %).

![MLR in four bins of calibrated metallicity](docs/mlr_feh_bins.png)

*The relation in four bins of the calibrated metallicity $Z$ (Fig. 6 of the paper). Upper panels: mass against $M_G$;
lower panels: difference from PARSEC at the same $Z$. Solid line and bands: this work (68 and 95 % credible
intervals); dashed line: PARSEC. Symbols are independent literature masses (Mann et al. 2019, Serenelli et al. 2017,
Xiong et al. 2023, Chevalier et al. 2023), shown on their own metallicity scales.*

## Install and use

```bash
pip install .            # numpy and scipy only
```

```python
import numpy as np
from wbmlr import MLR, calibrate_metallicity

mlr = MLR()
mg = np.array([5.0, 8.0, 11.0])
z = calibrate_metallicity(np.array([-0.1, 0.0, 0.1]), mg)   # raw J-CaPS [M/H] -> calibrated Z
lo, med, hi = mlr.mass_quantiles(mg, z)                     # 16, 50, 84 per cent, in Msun
ok = mlr.in_data_region(mg, z)

draws = mlr.mass_draws(mg, z)                               # (4000, 3) posterior draws
total = draws.sum(axis=1)                                   # correlated errors propagate correctly
```

`MLR.mass`, `mass_quantiles`, `mass_draws` and `log_mass_draws` take arrays of any shape (broadcast against each
other). See `examples/example.py`.

## Important: metallicity scale

`Z` is the calibrated metallicity on the PARSEC [M/H] scale, **not** the raw J-CaPS value. A raw Gaia XP metallicity
must be converted with `calibrate_metallicity(mh_jcaps, mg)`, which removes the magnitude-dependent J-CaPS bias
$b_G(M_G)$ and anchors the zero point to the Sun. For spectroscopic metallicities on another scale the conversion is
not defined, and the result is approximate.

## Data files (`src/wbmlr/data`)

| File | Content |
|---|---|
| `mlr_grid.csv` | Table on a $0.05$ mag $\times$ $0.05$ dex grid: `MG, Z, mass_p16, mass_p50, mass_p84` (Msun), `mass_parsec`, `delta_parsec_percent` (median relative to PARSEC at the same $Z$), `in_data_region` (0/1) |
| `mlr_theta_draws.npz` | 4000 posterior draws of the $8\times4$ B-spline coefficients `theta` of $\log_{10}(M/M_\odot)$, plus `knots_x`, `knots_z`, degrees. $g(M_G,Z)=\sum_{ih}\theta_{ih}B_i(M_G)C_h(Z)$, cubic, clamped knots |
| `data_coverage.npz` | Smoothed density of the sample components in $(M_G, Z)$ and the level enclosing 95 % of them |
| `jcaps_bias.csv` | Knots of the J-CaPS bias $b_G(M_G)$ (solar magnitude 4.66 for the zero point) |
| `meta.json` | Provenance |

## What the uncertainties contain

The quantiles are statistical, from the posterior of the adopted fit (eccentricity index $\alpha_e=1.233$, outlier
Gaussian with mean 42 and width 23 km s$^{-1}$ AU$^{1/2}$, selection cut $u/\sigma_u>3$). Further systematic effects
(see the paper): the shape of $p(\tilde u)$, $\alpha_e=1.0$–$1.5$, changes the masses by 0.4–1.9 %; fitting the
outlier mean and width as free parameters changes the masses by about 1 % at most but widens the uncertainty by
8–60 %. At $M_G>10.5$ the individual dynamical masses in the literature lie 8–10 % below this relation.

The relation describes single main-sequence stars. It does not apply to unresolved binaries, pre-main-sequence
stars or evolved stars.

## Tests

```bash
pip install ".[test]" && pytest
```

## Citation

Wang, Rix, Li & Ting (in prep.), *Dynamical calibration of a metallicity-dependent stellar mass–luminosity relation
from Gaia wide binaries*. Please also cite the J-CaPS metallicities (Li et al. 2026) if you use `calibrate_metallicity`.

The files in `src/wbmlr/data` were produced from the analysis repository by `scripts/export_public_mlr.py`
(https://github.com/yt-wangg/Dynamical-masses).

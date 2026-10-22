import numpy as np
from wbmlr import MLR

mlr = MLR()

# 1. a star on the colour-magnitude diagram: M_G and intrinsic (dereddened) BP-RP
mg = np.array([5.0, 8.0, 11.0])
bp_rp = np.array([0.87, 1.75, 2.70])
z, z_std = mlr.z_from_color(mg, bp_rp, return_std=True)   # mean and scatter of Z of the calibration stars there
lo, med, hi = mlr.mass_from_cmd(mg, bp_rp)             # 16, 50, 84 per cent, marginalised over Z
for a, b, zz, l, m, h in zip(mg, bp_rp, z, lo, med, hi):
    print(f"M_G={a:5.2f} BP-RP={b:4.2f} -> Z={zz:+.2f}: M = {m:.3f} (+{h-m:.3f} -{m-l:.3f}) Msun")

# 2. a star with a metallicity from your own measurement
print("M(M_G=7, Z=-0.2) =", mlr.mass(7.0, -0.2), "Msun; in data region:", bool(mlr.in_data_region(7.0, -0.2)))

# 3. total mass of a pair with propagated, correlated uncertainty
draws = mlr.mass_draws(np.array([5.0, 8.0]), 0.0)     # shape (4000, 2)
print("M1+M2 =", np.percentile(draws.sum(axis=1), [16, 50, 84]))

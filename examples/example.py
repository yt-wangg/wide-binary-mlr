import numpy as np
from wbmlr import MLR, calibrate_metallicity

mlr = MLR()
mg = np.array([5.0, 8.0, 11.0])
z = calibrate_metallicity(np.array([-0.1, 0.0, 0.1]), mg)      # raw J-CaPS [M/H] -> calibrated Z
lo, med, hi = mlr.mass_quantiles(mg, z)
for m, zz, a, b, c, ok in zip(mg, z, lo, med, hi, mlr.in_data_region(mg, z)):
    print(f"M_G={m:5.2f} Z={zz:+.2f}: M = {b:.3f} (+{c-b:.3f} -{b-a:.3f}) Msun   in data region: {bool(ok)}")

# total mass of a pair with propagated, correlated uncertainty
draws = mlr.mass_draws(np.array([5.0, 8.0]), 0.0)        # shape (4000, 2)
print("M1+M2 =", np.percentile(draws.sum(axis=1), [16, 50, 84]))

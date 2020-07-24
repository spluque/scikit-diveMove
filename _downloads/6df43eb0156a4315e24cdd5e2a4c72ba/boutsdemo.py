# Set up
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skdiveMove.tests import diveMove2skd
import skdiveMove.bouts as skbouts

# For figure sizes
_FIG1X1 = (7, 6)
_FIG1X2 = (12, 5)
_FIG3X1 = (11, 11)

pd.set_option("display.precision", 3)
np.set_printoptions(precision=3, sign="+")
%matplotlib inline

tdrX = diveMove2skd()
pars = {"offset_zoc": 3,
        "dry_thr": 70,
        "wet_thr": 3610,
        "dive_thr": 3,
        "dive_model": "unimodal",
        "smooth_par": 0.1,
        "knot_factor": 20,
        "descent_crit_q": 0.01,
        "ascent_crit_q": 0}

tdrX.calibrate(zoc_method="offset", offset=pars["offset_zoc"],
               dry_thr=pars["dry_thr"], wet_thr=pars["dry_thr"],
               dive_thr=pars["dive_thr"],
               dive_model=pars["dive_model"],
               smooth_par=pars["smooth_par"],
               knot_factor=pars["knot_factor"],
               descent_crit_q=pars["descent_crit_q"],
               ascent_crit_q=pars["ascent_crit_q"])
stats = tdrX.dive_stats()
stamps = tdrX.stamp_dives(ignore_z=True)
stats_tab = pd.concat((stamps, stats), axis=1)
stats_tab.info()

postdives = stats_tab["postdive_dur"][stats_tab["phase_id"] == 4]
postdives_diff = postdives.dt.total_seconds().diff()[1:].abs()
# Remove isolated dives
postdives_diff = postdives_diff[postdives_diff < 2000]

postdives_nlsbouts = skbouts.BoutsNLS(postdives_diff, 0.1)
print(postdives_nlsbouts)

fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars2 = postdives_nlsbouts.init_pars([50], plot=True, ax=ax)

coefs2, pcov2 = postdives_nlsbouts.fit(init_pars2)
# Coefficients
print(coefs2)

# Covariance between parameters
print(pcov2)

# `bec` returns ndarray, and we have only one here
print("bec = {[0]:.2f}".format(postdives_nlsbouts.bec(coefs2)))

fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_nlsbouts.plot_fit(coefs2, ax=ax);

fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars3 = postdives_nlsbouts.init_pars([50, 550], plot=True, ax=ax)

coefs3, pcov3 = postdives_nlsbouts.fit(init_pars3)
# Coefficients
print(coefs3)

# Covariance between parameters
print(pcov3)

fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_nlsbouts.plot_fit(coefs3, ax=ax);

fig, axs = plt.subplots(1, 2, figsize=_FIG1X2)
postdives_nlsbouts.plot_ecdf(coefs2, ax=axs[0])
postdives_nlsbouts.plot_ecdf(coefs3, ax=axs[1]);

postdives_mlebouts = skbouts.BoutsMLE(postdives_diff, 0.1)
print(postdives_mlebouts)

fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars = postdives_mlebouts.init_pars([50], plot=True, ax=ax)

p_bnd = (-2, None)                 # bounds for `p`
lda1_bnd = (-5, None)              # bounds for `lambda1`
lda2_bnd = (-10, None)             # bounds for `lambda2`
bnd1 = (p_bnd, lda1_bnd, lda2_bnd)
p_bnd = (1e-8, None)
lda1_bnd = (1e-8, None)
lda2_bnd = (1e-8, None)
bnd2 = (p_bnd, lda1_bnd, lda2_bnd)
fit1, fit2 = postdives_mlebouts.fit(init_pars,
                                    fit1_opts=dict(method="L-BFGS-B",
                                                   bounds=bnd1),
                                    fit2_opts=dict(method="L-BFGS-B",
                                                   bounds=bnd2))

# First fit
print(fit1)

# Second fit
print(fit2)

print("bec = {:.2f}".format(postdives_mlebouts.bec(fit2)))

fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_mlebouts.plot_fit(fit2, ax=ax);

fig, axs = plt.subplots(1, 2, figsize=_FIG1X2)
postdives_nlsbouts.plot_ecdf(coefs2, ax=axs[0])
axs[0].set_title("NLS")
postdives_mlebouts.plot_ecdf(fit2, ax=axs[1])
axs[1].set_title("MLM");

bec = postdives_mlebouts.bec(fit2)
skbouts.label_bouts(postdives.dt.total_seconds(), bec, as_diff=True)
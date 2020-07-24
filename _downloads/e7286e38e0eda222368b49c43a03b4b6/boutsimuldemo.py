# Set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skdiveMove.bouts as skbouts

# For figure sizes
_FIG3X1 = (9, 12)

pd.set_option("display.precision", 3)
np.set_printoptions(precision=3, sign="+")
%matplotlib inline

def genx(n, p, lda0, lda1):
    chooser = np.random.uniform(size=n)
    # We need to convert from scale to rate parameter
    proc1 = np.random.exponential(1 / lda0, size=n)
    proc2 = np.random.exponential(1 / lda1, size=n)
    proc_mix = np.where(chooser < p, proc1, proc2)
    return(proc_mix)

p_true = 0.7
lda0_true = 0.05
lda1_true = 0.005
pars_true = pd.Series({"lambda0": lda0_true,
                       "lambda1": lda1_true,
                       "p": p_true})

# Number of simulations
nsims = 500
# Size of each sample
nsamp = 1000

# Set up NLS simulations
coefs_nls = []
# Set up MLE simulations
coefs_mle = []
# Fixed bounds fit 1
p_bnd = (-2, None)
lda0_bnd = (-5, None)
lda1_bnd = (-10, None)
opts1 = dict(method="L-BFGS-B",
             bounds=(p_bnd, lda0_bnd, lda1_bnd))
# Fixed bounds fit 2
p_bnd = (1e-1, None)
lda0_bnd = (1e-3, None)
lda1_bnd = (1e-6, None)
opts2 = dict(method="L-BFGS-B",
             bounds=(p_bnd, lda0_bnd, lda1_bnd))

# Estimate parameters `nsims` times
for i in range(nsims):
    x = genx(nsamp, pars_true["p"], pars_true["lambda0"],
             pars_true["lambda1"])
    # NLS
    xbouts = skbouts.BoutsNLS(x, 5)
    init_pars = xbouts.init_pars([80], plot=False)
    coefs, _ = xbouts.fit(init_pars)
    p_i = skbouts.bouts.calc_p(coefs)[0][0]  # only one here
    coefs_i = coefs.loc["lambda"].append(pd.Series({"p": p_i}))
    coefs_nls.append(coefs_i.to_numpy())

    # MLE
    xbouts = skbouts.BoutsMLE(x, 5)
    init_pars = xbouts.init_pars([80], plot=False)
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=opts1,
                            fit2_opts=opts2)
    coefs_mle.append(np.roll(fit2.x, -1))

nls_coefs = pd.DataFrame(np.row_stack(coefs_nls),
                         columns=["lambda0", "lambda1", "p"])
# Centrality and variance
nls_coefs.describe()

mle_coefs = pd.DataFrame(np.row_stack(coefs_mle),
                         columns=["lambda0", "lambda1", "p"])
# Centrality and variance
mle_coefs.describe()

nls_coefs.mean() - pars_true

mle_coefs.mean() - pars_true

# Combine results
coefs_merged = pd.concat((mle_coefs, nls_coefs), keys=["mle", "nls"],
                         names=["method", "idx"])

# Density plots
kwargs = dict(alpha=0.8)
fig, axs = plt.subplots(3, 1, figsize=_FIG3X1)
lda0 = (coefs_merged["lambda0"].unstack(level=0)
        .plot(ax=axs[0], kind="kde", legend=False, **kwargs))
axs[0].set_ylabel(r"Density $[\lambda_f]$")
# True value
axs[0].axvline(pars_true["lambda0"], linestyle="dashed", color="k")
lda1 = (coefs_merged["lambda1"].unstack(level=0)
        .plot(ax=axs[1], kind="kde", legend=False, **kwargs))
axs[1].set_ylabel(r"Density $[\lambda_s]$")
# True value
axs[1].axvline(pars_true["lambda1"], linestyle="dashed", color="k")
p_coef = (coefs_merged["p"].unstack(level=0)
          .plot(ax=axs[2], kind="kde", legend=False, **kwargs))
axs[2].set_ylabel(r"Density $[p]$")
# True value
axs[2].axvline(pars_true["p"], linestyle="dashed", color="k")
axs[0].legend(["MLE", "NLS"], loc=8, bbox_to_anchor=(0.5, 1),
              frameon=False, borderaxespad=0.1, ncol=2);
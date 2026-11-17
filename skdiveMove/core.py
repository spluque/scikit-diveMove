"""Lazy access to the R objects used by R-backed operations.

"""

from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize=1)
def get_r():
    """Initialize and return the shared R objects on first use."""
    import rpy2.robjects as robjs
    import rpy2.robjects.conversion as cv
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    return SimpleNamespace(
        robjs=robjs,
        cv=cv,
        pandas2ri=pandas2ri,
        diveMove=importr("diveMove"),
        r_base=importr("base"),
    )

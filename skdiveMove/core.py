"""Centralized rpy2 imports to share a single instance in submodules

"""

from rpy2.robjects.packages import importr  # noqa: F401
import rpy2.robjects as robjs               # noqa: F401
import rpy2.robjects.conversion as cv       # noqa: F401
from rpy2.robjects import pandas2ri         # noqa: F401

# Initialize R instance
diveMove = importr("diveMove")
# For accessing R base objects
r_base = importr("base")

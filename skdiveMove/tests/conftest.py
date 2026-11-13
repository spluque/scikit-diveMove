"""Shared pytest fixtures for scikit-diveMove tests.

Provides reusable fixtures for test data, directory management, matplotlib
backend configuration, and sample data loading.

"""

import pytest
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")

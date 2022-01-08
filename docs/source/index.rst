=================================
 scikit-diveMove Documentation
=================================

`scikit-diveMove` is a Python interface to R package `diveMove`_ for
scientific data analysis, with a focus on diving behaviour analysis.  It
has utilities to represent, visualize, filter, analyse, and summarize
time-depth recorder (TDR) data.  Miscellaneous functions for handling
location data are also provided.

.. _diveMove: https://github.com/spluque/diveMove

`scikit-diveMove` is hosted at https://github.com/spluque/scikit-diveMove

`scikit-diveMove` also provides useful tools for processing signals from
tri-axial Inertial Measurement Units (`IMU`_), such as thermal calibration,
corrections for shifts in coordinate frames, as well as computation of
orientation using a variety of current methods.  Analyses are fully
tractable by encouraging the use of `xarray`_ data structures that can be
read from and written to NetCDF file format.  Using these data structures,
meta-data attributes can be easily appended at all layers as analyses
progress.

.. _xarray: https://xarray.pydata.org
.. _IMU: https://en.wikipedia.org/wiki/Inertial_measurement_unit


Installation
============

Type the following at a terminal command line:

.. code-block:: sh

   pip install scikit-diveMove

Or install from source tree by typing the following at the command line:

.. code-block:: sh

   python setup.py install

Once installed, `skdiveMove` can be easily imported as: ::

  import skdiveMove as skdive


Dependencies
------------

`skdiveMove` depends primarily on ``R`` package `diveMove`, which must be
installed and available to the user running Python.  If needed, install
`diveMove` at the ``R`` prompt:

.. code-block:: R

   install.packages("diveMove")

Required Python packages are listed in the `requirements`_ file.

.. _requirements: https://github.com/spluque/scikit-diveMove/blob/master/requirements.txt


Testing
=======

The `skdiveMove` package can be tested with `unittest`:

.. code-block:: sh

   python -m unittest -v skdiveMove/tests

or `pytest`:

.. code-block:: sh

   pytest -v skdiveMove/tests


Development
===========

Developers can clone the project from Github:

.. code-block:: sh

   git clone https://github.com/spluque/scikit-diveMove.git .

and then install with:

.. code-block:: sh

   pip install -e .["dev"]


Demos
=====

.. toctree::
   :maxdepth: 2

   demo_tdr
   demo_bouts
   demo_simulbouts
   imutools_demos


Modules
=======

.. toctree::
   :maxdepth: 2

   tdr
   bouts
   metadata
   imutools


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

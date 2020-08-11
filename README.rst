.. raw:: html

   <img alt="scikit-diveMove" src="docs/source/.static/skdiveMove_logo.png"
    width=10% align=left>
   <h1>scikit-diveMove</h1>

.. image:: https://img.shields.io/pypi/v/scikit-diveMove
   :target: https://pypi.python.org/pypi/scikit-diveMove
   :alt: PyPI

.. image:: https://github.com/spluque/scikit-diveMove/workflows/TestPyPI/badge.svg
   :target: https://github.com/spluque/scikit-diveMove/actions?query=workflow%3ATestPyPI
   :alt: TestPyPI

.. image:: https://github.com/spluque/scikit-diveMove/workflows/Python%20build/badge.svg
   :target: https://github.com/spluque/scikit-diveMove/actions?query=workflow%3A%22Python+build%22
   :alt: Python Build

.. image:: https://codecov.io/gh/spluque/scikit-diveMove/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/spluque/scikit-diveMove

.. image:: https://img.shields.io/pypi/dm/scikit-diveMove
   :target: https://pypi.python.org/pypi/scikit-diveMove
   :alt: PyPI - Downloads


`scikit-diveMove` is a Python interface to R package `diveMove`_ for
scientific data analysis, with a focus on diving behaviour analysis.  It
has utilities to represent, visualize, filter, analyse, and summarize
time-depth recorder (TDR) data.  Miscellaneous functions for handling
location data are also provided.  `scikit-diveMove` communicates with a
single `R` instance for access to low-level tools of package `diveMove`.

.. _diveMove: https://github.com/spluque/diveMove

The table below shows which features of `diveMove` are accessible from
`scikit-diveMove`:

+----------------------------------+--------------------------+--------------------------------+
|                  `diveMove`      |`scikit-diveMove`         |Notes                           |
+---------------+------------------+                          |                                |
|Functionality  |Functions/Methods |                          |                                |
+===============+==================+==========================+================================+
|Movement       |``austFilter``    |                          |Under consideration.            |
|               |``rmsDistFilter`` |                          |                                |
|               |``grpSpeedFilter``|                          |                                |
|               |``distSpeed``     |                          |                                |
|               |``readLocs``      |                          |                                |
+---------------+------------------+--------------------------+--------------------------------+
|Bout analysis  |``boutfreqs``     |``BoutsNLS`` ``BoutsMLE`` |Fully implemented in Python.    |
|               |``boutinit``      |                          |                                |
|               |``bouts2.nlsFUN`` |                          |                                |
|               |``bouts2.nls``    |                          |                                |
|               |``bouts3.nlsFUN`` |                          |                                |
|               |``bouts3.nls``    |                          |                                |
|               |``bouts2.mleFUN`` |                          |                                |
|               |``bouts2.ll``     |                          |                                |
|               |``bouts2.LL``     |                          |                                |
|               |``bouts.mle``     |                          |                                |
|               |``labelBouts``    |                          |                                |
|               |``plotBouts``     |                          |                                |
|               |``plotBouts2.cdf``|                          |                                |
|               |``bec2``          |                          |                                |
|               |``bec3``          |                          |                                |
+---------------+------------------+--------------------------+--------------------------------+
|Dive analysis  |``readTDR``       |``TDR.__init__``          |Fully implemented.  Single      |
|               |``createTDR``     |``TDRSource.__init__``    |``TDR`` class for data with or  |
|               |                  |                          |without speed measurements.     |
+---------------+------------------+--------------------------+--------------------------------+
|               |``calibrateDepth``|``TDR.calibrate``         |Fully implemented               |
|               |                  |``TDR.zoc``               |                                |
|               |                  |``TDR.detect_wet``        |                                |
|               |                  |``TDR.detect_dives``      |                                |
|               |                  |``TDR.detect_dive_phases``|                                |
+---------------+------------------+--------------------------+--------------------------------+
|               |``calibrateSpeed``|``TDR.calibrate_speed``   |New implementation of the       |
|               |``rqPlot``        |                          |algorithm entirely in Python.   |
|               |                  |                          |The procedure generates the plot|
|               |                  |                          |concurrently.                   |
+---------------+------------------+--------------------------+--------------------------------+
|               |``diveStats``     |``TDR.dive_stats``        |Fully implemented               |
|               |``stampDive``     |``TDR.time_budget``       |                                |
|               |``timeBudget``    |``TDR.stamp_dives``       |                                |
+---------------+------------------+--------------------------+--------------------------------+
|               |``plotTDR``       |``TDR.plot``              |Fully implemented.              |
|               |``plotDiveModel`` |``TDR.plot_zoc_filters``  |Interactivity is the default, as|
|               |``plotZOC``       |``TDR.plot_phases``       |standard `matplotlib`.          |
|               |                  |``TDR.plot_dive_model``   |                                |
+---------------+------------------+--------------------------+--------------------------------+
|               |``getTDR``        |``TDR.tdr``               |Fully implemented.              |
|               |``getDepth``      |``TDR.get_depth``         |``getCCData`` deemed redundant, |
|               |``getSpeed``      |``TDR.get_speed``         |as the columns can be accessed  |
|               |``getTime``       |``TDR.tdr.index``         |directly from the ``TDR.tdr``   |
|               |``getCCData``     |``TDR.src_file``          |attribute.                      |
|               |``getDtime``      |``TDR.dtime``             |                                |
|               |``getFileName``   |                          |                                |
+---------------+------------------+--------------------------+--------------------------------+
|               |``getDAct``       |``TDR.get_wet_activity``  |Fully implemented               |
|               |``getDPhaseLab``  |``TDR.get_dives_details`` |                                |
|               |``getDiveDeriv``  |``TDR.get_dive_deriv``    |                                |
|               |``getDiveModel``  |                          |                                |
|               |``getGAct``       |                          |                                |
+---------------+------------------+--------------------------+--------------------------------+
|               |``extractDive``   |                          |Fully implemented               |
+---------------+------------------+--------------------------+--------------------------------+


Installation
============

Type the following at a terminal command line:

.. code-block:: sh

   pip install scikit-kinematics

Or install from source tree by typing the following at the command line:

.. code-block:: sh

   python setup.py install

The documentation can also be installed as described in `Documentation`_.

Once installed, `skdiveMove` can be easily imported as: ::

  import skdiveMove as skdive


Dependencies
------------

`skdiveMove` depends primarily on ``R`` package `diveMove`, which must be
installed and available to the user running Python.  If needed, install
`diveMove` at the ``R`` prompt:

.. code-block:: R

   install.packages("diveMove")

Required Python packages are listed in the `requirements
<requirements.txt>`_ file.


Documentation
=============

Available at: https://spluque.github.io/scikit-diveMove

Alternatively, installing the package as follows:

.. code-block:: sh

   pip install -e .["docs"]

allows the documentation to be built locally (choosing the desired target
{"html", "pdf", etc.}):

.. code-block:: sh

   make -C docs/ html

The `html` tree is at `docs/build/html`.

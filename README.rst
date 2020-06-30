.. raw:: html

   <img alt="scikit-diveMove" src="docs/source/.static/skdiveMove_logo.png"
    width=10% align=left>
   <h1>scikit-diveMove</h1>

.. image:: https://travis-ci.org/spluque/scikit-diveMove.svg?branch=master
   :target: https://travis-ci.org/spluque/scikit-diveMove
   :alt: Build Status


`scikit-diveMove` is a Python interface to R package `diveMove`_ for
scientific data analysis, with a focus on diving behaviour analysis.  It
has utilities to represent, visualize, filter, analyse, and summarize
time-depth recorder (TDR) data.  Miscellaneous functions for handling
location data are also provided.

.. _diveMove: https://github.com/spluque/diveMove


Installation
============

At some point, it will be possible to install `skdiveMove` by typing the
following at a terminal command line:

.. code-block:: sh

   pip install scikit-kinematics

In the meantime, please install from the source files by typing the
following at the command line:

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

The following Python packages must be installed:

  - `docutils` (>= 0.3)
  - `matplotlib` (> 3.0)
  - `numpy` (>= 1.18)
  - `pandas` (>= 1.0)
  - `rpy2` (>= 3.3)


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

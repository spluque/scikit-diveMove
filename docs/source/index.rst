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


Installation
============

Type the following at a terminal command line:

.. code-block:: sh

   pip install scikit-kinematics

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

Developers and hackers can clone the project from Github:

.. code-block:: sh

   git clone https://github.com/spluque/scikit-diveMove.git .

and then install with:

.. code-block:: sh

   pip install -e .["dev"]


Demos
=====

.. toctree::
   :maxdepth: 2

   tdrdemo
   boutsdemo
   boutsimuldemo


Modules
=======

.. toctree::
   :maxdepth: 2

   tdr
   tdrsource
   zoc
   tdrphases
   bouts


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

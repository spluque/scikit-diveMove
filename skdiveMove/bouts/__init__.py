r"""Tools and classes for the identification of behavioural bouts

A histogram of log-transformed frequencies of `x` with a chosen bin width
and upper limit forms the basis for models.  Histogram bins following empty
ones have their frequencies averaged over the number of previous empty bins
plus one.  Models attempt to discern the number of random Poisson
processes, and their parameters, generating the underlying distribution of
log-transformed frequencies.

The abstract class :class:`Bouts` provides basic methods.

Abstract class & methods summary
--------------------------------

.. autosummary::

   Bouts
   Bouts.init_pars
   Bouts.fit
   Bouts.bec
   Bouts.plot_fit


Nonlinear least squares models
------------------------------

Currently, the model describing the histogram as it is built is implemented
in the :class:`BoutsNLS` class.  For the case of a mixture of two Poisson
processes, this class would set up the model:

.. math::
   :label: 1

   y = log[N_f \lambda_f  e^{-\lambda_f  t} +
           N_s \lambda_s e^{-\lambda_s t}]

where :math:`N_f` and :math:`N_s` are the number of events belonging to
process :math:`f` and :math:`s`, respectively; and :math:`\lambda_f` and
:math:`\lambda_s` are the probabilities of an event occurring in each
process.  Mixtures of more processes can also be added to the model.

The bout-ending criterion (BEC) corresponding to equation :eq:`1` is:

.. math::
   :label: 2

   BEC = \frac{1}{\lambda_f - \lambda_s}
         log \frac{N_f \lambda_f}{N_s \lambda_s}

Note that there is one BEC per transition between Poisson processes.

The methods of this subclass are provided by the abstract super class
:class:`Bouts`, and defining those listed below.

Methods summary
---------------

.. autosummary::

   BoutsNLS.plot_ecdf


Maximum likelihood models
-------------------------

This is the preferred approach to modelling mixtures of random Poisson
processes, as it does not rely on the subjective construction of a
histogram.  The histogram is only used to generate reasonable starting
values, but the underlying paramters of the model are obtained via maximum
likelihood, so it is more robust.

For the case of a mixture of two processes, as above, the log likelihood of
all the :math:`N_t` in a mixture can be expressed as:

.. math::
   :label: 3

   log\ L_2 = \sum_{i=1}^{N_t} log[p \lambda_f e^{-\lambda_f t_i} +
                                   (1-p) \lambda_s e^{-\lambda_s t_i}]

where :math:`p` is a mixing parameter indicating the proportion of fast to
slow process events in the sampled population.

The BEC in this case can be estimated as:

.. math::
   :label: 4

   BEC = \frac{1}{\lambda_f - \lambda_s}
         log \frac{p\lambda_f}{(1-p)\lambda_s}

The subclass :class:`BoutsMLE` offers the framework for these models.

Class & methods summary
-----------------------

.. autosummary::

   BoutsMLE.negMLEll
   BoutsMLE.fit
   BoutsMLE.bec
   BoutsMLE.plot_fit
   BoutsMLE.plot_ecdf


API
---

"""

from .bouts import Bouts, label_bouts
from .boutsnls import BoutsNLS
from .boutsmle import BoutsMLE
from skdiveMove.tests import random_mixexp

__all__ = ["Bouts", "BoutsNLS", "BoutsMLE", "label_bouts",
           "random_mixexp"]

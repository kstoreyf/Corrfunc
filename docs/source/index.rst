******************************************
`suave`: The Continuous-Function Estimator
******************************************

Introduction
============

The suave package is an implementation generalized estimator of the two-point correlation function (2pcf) for cosmology. The 2pcf is the most important statistic for the analysis of large-scale structure; it measures the strength of clustering (e.g. of galaxies) as a function of separation. Suave replaces the standard binning in separation with a projection of galaxy pair counts onto any set of basis functions. The choice of basis functions can preserve more information for improved bias and variance properties, include other galaxy information, and be specific to the science use case.

The source code is publicly available at https://github.com/kstoreyf/suave. The paper is available on the arXiv at https://arxiv.org/abs/2011.01836.

This implementation of the suave estimator is built within and on top of the Corrfunc package (https://github.com/manodeep/Corrfunc). As such, all of the Corrfunc functionality is accessible through suave. Here we only document the new and updated functionality of suave; for full Corrfunc usage, see https://corrfunc.readthedocs.io.


The basics
============

.. toctree::
   :maxdepth: 1

   install-suave
   ./basic_usage.nblink


Demonstrations
==============

.. toctree::
   :maxdepth: 1

   ./example_theory.nblink
   ./example_bao_theory.nblink


API Reference
=============

.. toctree::
   :maxdepth: 4
   
   api/modules-suave
    

The nitty-gritty
===================

.. toctree::
   :maxdepth: 1

   development/dev-suave


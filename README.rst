`suave`: The Continuous-Function Estimator
==========================================

|PyPI| |MIT licensed| |RTD| |Continuous-Function Estimator Paper| |Corrfunc Paper I| |Corrfunc Paper II|

This is an implementation of the Continuous-Function Estimator, a generalization of the standard (Landy-Szalay) estimator for the two-point correlation function. 
We call this tool ``suave`` which means *smooth* in Spanish (pronounced **swah**-beh), as it can produce smooth (continuous) correlation functions. 
It is built within the `Corrfunc <https://github.com/manodeep/Corrfunc>`_  package, by Manodeep Sinha and Lehman Garrison; check out the full Corrfunc README at the original repo.

The 2-point correlation function measures the clustering of galaxies (or other tracers) as a function of scale. 
Traditionally, this is done by counting the pairs of galaxies in a given separation bin, and normalizing by the pairs in a uniform random catalog. 

The Continuous-Function Estimator eliminates the need for binning, in separation or any other quantity. 
Rather, it projects the pairs onto any user-defined set of basis functions.
It replaces the pair counts with vectors, and the random normalization vector term with a matrix, that describe the contribution of the pairs to each basis function.
The correlation function can then be directly evaluated at any separation, resulting in a continuous estimation.

An example script for using the estimator is in `example_theory.ipynb <https://github.com/kstoreyf/Corrfunc/blob/master/examples/example_theory.ipynb>`_.
The Continuous-Function Estimator is currently implemented in the DD(s, mu) pair counting statistic for both mock and theory data.
Currently implemented bases are tophat and piecewise.
General r-dependent basis functions can be read in from a file; helper routines for these include spline basis functions of any order and a baryon acoustic oscillation fitting function.

The paper presenting this method can be found at https://arxiv.org/abs/2011.01836 (Storey-Fisher \& Hogg, Accepted to ApJ). 
Feel free to email `k.sf@nyu.edu <mailto:k.sf@nyu.edu>`_ with any comments or questions, or `submit an issue <https://github.com/kstoreyf/Corrfunc/issues/new/choose>`_.

Installation
============

Pre-requisites
--------------

Suave has most of the same pre-reqs as Corrfunc, as well as a couple more:

- ``make >= 3.80``
- OpenMP capable compiler like ``icc``, ``gcc>=4.6`` or ``clang >= 3.7``. You should already have a system install, but on mac/linux you can install gcc with ``conda install gcc``.
- ``gsl >= 2.4``. Use either ``conda install -c conda-forge gsl`` (MAC/linux) or ``(sudo) port install gsl`` (MAC) to install ``gsl`` if necessary.
- ``python >= 2.7`` or ``python>=3.4`` for compiling the C extensions.
- ``numpy >= 1.7`` for compiling the C extensions.
- ``scipy >= 1.6`` for the spline basis functions for suave (lower versions may work but untested) 
- ``colossus >= 1.2`` for the BAO basis functions for suave (lower versions may work but untested)  
- ``six >= 1.15`` (colossus dependency, lower versions may work but untested)

Install with pip
----------------

You can install suave via pip. We recommend doing this into a clean conda environment. You can do this and install the dependencies with the following set of commands:

::

   $ conda create -n suaveenv
   $ conda activate suaveenv
   $ conda install -c conda-forge gsl numpy scipy six pip
   $ pip install colossus
   $ pip install suave

Install from source
-------------------

You should also be able to install from source. Once again you can do this in a clean conda environment:

::

   $ conda create -n suaveenv
   $ conda activate suaveenv
   $ conda install -c conda-forge gsl numpy scipy six pip
   $ pip install colossus
   $ git clone https://github.com/kstoreyf/suave/
   $ cd suave
   $ make
   $ make install
   $ pip install . (--user)

Author & Maintainers
====================

The suave package was implemented by `Kate Storey-Fisher <https://github.com/kstoreyf>`_.
It is built within Corrfunc, which was designed by Manodeep Sinha and is currently maintained by
`Lehman Garrison <https://github.com/lgarrison>`_ and `Manodeep Sinha <https://github.com/manodeep>`_.

Citing
======

If you use or reference ``suave``, please cite the ApJ paper with this bibtex entry (this will be updated once the accepted paper is published):

::

   @misc{storeyfisher2020twopoint,
      title={Two-point statistics without bins: A continuous-function generalization of the correlation function estimator for large-scale structure}, 
      author={Kate Storey-Fisher and David W. Hogg},
      year={2020},
      eprint={2011.01836},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO}
   }


If you use the code, please additionally cite the original MNRAS ``Corrfunc`` code paper with the following
bibtex entry:

::

   @ARTICLE{2020MNRAS.491.3022S,
       author = {{Sinha}, Manodeep and {Garrison}, Lehman H.},
       title = "{CORRFUNC - a suite of blazing fast correlation functions on
       the CPU}",
       journal = {\mnras},
       keywords = {methods: numerical, galaxies: general, galaxies:
       haloes, dark matter, large-scale structure of Universe, cosmology:
       theory},
       year = "2020",
       month = "Jan",
       volume = {491},
       number = {2},
       pages = {3022-3041},
       doi = {10.1093/mnras/stz3157},
       adsurl =
       {https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3022S},
       adsnote = {Provided by the SAO/NASA
       Astrophysics Data System}
   }


Finally, if you benefit from the enhanced vectorised kernels in ``Corrfunc`` (not currently used in ``suave`` but likely used if you're also using out-of-the-box ``Corrfunc``), then please also cite this paper:

::

      @InProceedings{10.1007/978-981-13-7729-7_1,
          author="Sinha, Manodeep and Garrison, Lehman",
          editor="Majumdar, Amit and Arora, Ritu",
          title="CORRFUNC: Blazing Fast Correlation Functions with AVX512F SIMD Intrinsics",
          booktitle="Software Challenges to Exascale Computing",
          year="2019",
          publisher="Springer Singapore",
          address="Singapore",
          pages="3--20",
          isbn="978-981-13-7729-7",
          url={https://doi.org/10.1007/978-981-13-7729-7_1}
      }

LICENSE
=======

Suave is released under the MIT license. Basically, do what you want
with the code, including using it in commercial application.

Project URLs
============

-  Documentation (http://suave.rtfd.io/)
-  Source Repository (https://github.com/kstoreyf/suave)
-  Original Corrfunc Documentation (http://corrfunc.rtfd.io/)
-  Original Corrfunc Source Repository (https://github.com/manodeep/Corrfunc)

Support
=======

This work was supported by a NASA FINESST grant under award 80NSSC20K1545.


.. |logo| image:: https://github.com/manodeep/Corrfunc/blob/master/corrfunc_logo.png
    :target: https://github.com/manodeep/Corrfunc
    :alt: Corrfunc logo
.. |Release| image:: https://img.shields.io/github/release/kstoreyf/suave.svg
   :target: https://github.com/kstoreyf/suave/releases/latest
   :alt: Latest Release
.. |PyPI| image:: https://img.shields.io/pypi/v/suave.svg
   :target: https://pypi.python.org/pypi/suave
   :alt: PyPI Release
.. |MIT licensed| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://raw.githubusercontent.com/manodeep/Corrfunc/master/LICENSE
   :alt: MIT License
.. |RTD| image:: https://readthedocs.org/projects/suave/badge/?version=master
   :target: http://suave.readthedocs.io/en/master/?badge=master
   :alt: Documentation Status

.. |Continuous-Function Estimator Paper| image:: https://img.shields.io/badge/arXiv-2011.01836-%23B31B1B
   :target: https://arxiv.org/abs/2011.01836
   :alt: Continuous-Function Estimator Paper
.. |Corrfunc Paper I| image:: https://img.shields.io/badge/arXiv-1911.03545-%23B31B1B
   :target: https://arxiv.org/abs/1911.03545
   :alt: Corrfunc Paper I
.. |Corrfunc Paper II| image:: https://img.shields.io/badge/arXiv-1911.08275-%23B31B1B
   :target: https://arxiv.org/abs/1911.08275
   :alt: Corrfunc Paper II

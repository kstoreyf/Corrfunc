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

.. code::

   $ conda create -n suaveenv
   $ conda activate suaveenv
   $ conda install -c conda-forge gsl numpy scipy six pip
   $ pip install colossus
   $ pip install suave

Install from source
-------------------

You should also be able to install from source. Once again you can do this in a clean conda environment:

.. code::

   $ conda create -n suaveenv
   $ conda activate suaveenv
   $ conda install -c conda-forge gsl numpy scipy six pip
   $ pip install colossus
   $ git clone https://github.com/kstoreyf/suave/
   $ cd suave
   $ make
   $ make install
   $ pip install . (--user)
========================================
suave: The Continuous-Function Estimator
========================================

************
Introduction
************

The suave package is an implementation generalized estimator of the two-point correlation function (2pcf) for cosmology. The 2pcf is the most important statistic for the analysis of large-scale structure; it measures the strength of clustering (e.g. of galaxies) as a function of separation. Suave replaces the standard binning in separation with a projection of galaxy pair counts onto any set of basis functions. The choice of basis functions can preserve more information for improved bias and variance properties, include other galaxy information, and be specific to the science use case.

The source code is publicly available at https://github.com/kstoreyf/suave. The paper is available on the arXiv at https://arxiv.org/abs/2011.01836.

This implementation of the suave estimator is built on top of the Corrfunc package (https://github.com/manodeep/Corrfunc). As such, all of the Corrfunc functionality is accessible through suave. Here we only document the new and updated functionality of suave; for full Corrfunc usage, see https://corrfunc.readthedocs.io.


***********
Basic Usage
***********

Here we demonstrate the basic usage of the estimator by computing a typical correlation function with tophat bins, but in a continuous basis.

Load in a data catalog (in a periodic cube) and generate a random catalog::

    boxsize = 750.0
    x, y, z = read_lognormal_catalog(n='3e-4')
    nd = len(x)
    nr = 3*nd
    x_rand = np.random.uniform(0, boxsize, nr)
    y_rand = np.random.uniform(0, boxsize, nr)
    z_rand = np.random.uniform(0, boxsize, nr)

To use the tophat basis, set the `proj_type` accordingly and choose radial separation bins. To compute the full 3D correlation function, we can use the 2D correlation function DD(s, \mu) with a single \mu bin::

    proj_type = 'tophat'
    rmin, rmax, ncomponents = 40.0, 150.0, 22
    r_edges = np.linspace(rmin, rmax, ncomponents+1)
    nmubins = 1
    mumax = 1.0
    periodic = True
    nthreads = 1

Compute the component vectors DD, DR, and RR, which in this case are equivalent to the pair counts, and the component tensor T_RR::

    dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, 
                               x, y, z, boxsize=boxsize, periodic=periodic, 
                               proj_type=proj_type, ncomponents=ncomponents)
    dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, 
                               x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand,
                               boxsize=boxsize, periodic=periodic,
                               proj_type=proj_type, ncomponents=ncomponents)
    rr_res, rr_proj, trr_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, 
                                     x_rand, y_rand, z_rand, boxsize=boxsize,
                                     periodic=periodic, proj_type=proj_type,
                                     ncomponents=ncomponents)

From the component vectors, compute the amplitudes, and evaluate the correlation function on a fine r-grid::

    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
    r_fine = np.linspace(rmin, rmax, 2000)
    xi = evaluate_xi(amps, r_fine, proj_type, rbins=r_edges)


**************
Demo notebooks
**************

.. toctree::
   :maxdepth: 1

   ./example_theory.nblink


***************************************
API Reference: suave-specific functions 
***************************************

.. toctree::
   :maxdepth: 4
   
   api/modules-suave
    

*********************
License and Credits
*********************

.. toctree::
   :maxdepth: 1

   development/dev-suave


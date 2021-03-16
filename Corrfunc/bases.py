import numpy as np
from scipy.interpolate import BSpline
from colossus.cosmology import cosmology

"""
Helper routines for basis functions for the continuous-function estimator.
"""
################
# Spline basis #
################

def spline_bases(rmin, rmax, projfn, ncomponents, ncont=2000, order=3):
    '''
    Compute a set of spline basis functions for the given order.

    Parameters
    ----------
    rmin : double
        Minimum r-value for basis functions 

    rmax : double
        Maximum r-value for basis functions 

    projfn : string, default=None
       Path to projection file if necessary

    ncomponents : int
       Number of components (basis functions)

    ncont : int, default=2000
       Number of continuous r-values at which to write the basis function file

    order : int, default=3
       Order of spline to use; default is cubic spline

    Returns
    -------
    bases: array-like, double
        2-d array of basis function values; first column is r-values

    '''
    if ncomponents<order*2:
        raise ValueError("ncomponents must be at least twice the order")

    kvs = _get_knot_vectors(rmin, rmax, ncomponents, order)
    rcont = np.linspace(rmin, rmax, ncont)
    bases = np.empty((ncont, ncomponents+1))
    bases[:,0] = rcont
    for n in range(ncomponents):
        kv = kvs[n]
        b = BSpline.basis_element(kv)
        bases[:,n+1] = [b(r) if kv[0]<=r<=kv[-1] else 0 for r in rcont]

    np.savetxt(projfn, bases)
    return bases


def _get_knot_vectors(rmin, rmax, ncomponents, order):
    nknots = order+2
    kvs = np.empty((ncomponents, nknots))
    width = (rmax-rmin)/(ncomponents-order)
    for i in range(order):
        val = i+1
        kvs[i,:] = np.concatenate((np.full(nknots-val, rmin), np.linspace(rmin+width, rmin+width*val, val)))
        kvs[ncomponents-i-1] = np.concatenate((np.linspace(rmax-width*val, rmax-width, val), np.full(nknots-val, rmax)))
    for j in range(ncomponents-2*order):
        idx = j+order
        kvs[idx] = rmin+width*j + np.arange(0,nknots)*width                     
    return kvs


#############
# BAO basis #
#############

def bao_bases(rmin, rmax, projfn, cosmo_base=None, ncont=2000, 
              redshift=0.0, alpha_guess=1.0, dalpha=0.001, bias=1.0, 
              k0=0.1, k1=10.0, k2=0.1, k3=0.001):
    '''
    Compute the 5-component BAO basis functions based on a cosmological model and 
    linearized around the scale dilation parameter alpha.

    Parameters
    ----------
    rmin : double
        Minimum r-value for basis functions

    rmax : double
        Maximum r-value for basis functions 

    projfn : string, default=None
        Path to projection file if necessary

    cosmo_base : nbodykit cosmology object, default=nbodykit.cosmology.Planck15
        Cosmology object for the BAO model.

    ncont : int, default=2000
        Number of continuous r-values at which to write the basis function file

    redshift : double, default=0.0
        Redshift at which to compute power spectrum

    alpha_guess : double, default=1.0
        The alpha (scale dilation parameter) at which to compute the model (alpha=1.0 is no scale shift)

    dalpha : double, default=0.001
        The change in alpha (scale dilation parameter) used to calculate the numerical partial derivative
    
    bias : double, default=1.0
        The bias parameter by which to scale the model amplitude (bias=1.0 indicates no bias)

    k0 : double, default=0.1
        The initial magnitude of the derivative term 

    k1 : double, default=1.0
        The initial magnitude of the s^2 nuisance parameter term 

    k2 : double, default=0.1
        The initial magnitude of the s nuisance parameter term 

    k3 : double, default=0.001
        The initial magnitude of the constant nuisance parameter term 

    Returns
    -------
    bases: array-like, double
        2-d array of basis function values; first column is r-values

    '''
    if cosmo_base is None:
        print("cosmo_base not provided, defaulting to Planck 2015 cosmology ('planck15')")
        cosmo_base = cosmology.setCosmology('planck15')

    cf = cosmo_base.correlationFunction

    def cf_model(r):
        return bias * cf(r, z=redshift)

    rcont = np.linspace(rmin, rmax, ncont)
    bs = _get_bao_components(rcont, cf_model, dalpha, alpha_guess, k0=k0, k1=k1, k2=k2, k3=k3)

    nbases = len(bs)
    bases = np.empty((ncont, nbases+1))
    bases[:,0] = rcont
    bases[:,1:nbases+1] = np.array(bs).T

    np.savetxt(projfn, bases)
    ncomponents = bases.shape[1]-1
    return bases


def _get_bao_components(r, cf_func, dalpha, alpha, k0=0.1, k1=10.0, k2=0.1, k3=0.001):   
    b1 = k1/r**2
    
    b2 = k2/r

    b3 = k3*np.ones(len(r))
    
    cf = cf_func(alpha*r)
    b4 = cf

    cf_dalpha = cf_func((alpha+dalpha)*r)
    dcf_dalpha = _partial_derivative(cf, cf_dalpha, dalpha)
    b5 = k0*dcf_dalpha
    
    return b1,b2,b3,b4,b5
    

def _partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv    

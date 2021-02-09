import numpy as np
from scipy.interpolate import BSpline
from nbodykit.lab import cosmology

'''
Helper routines for basis functions
'''

################
# Spline basis #
################

def spline_bases(rmin, rmax, nbins, projfn, ncont=1000, order=None):

    if order is None:
        print("No order given, defaulting to 1 (linear)")
    
    if nbins<order*2:
        # does it have to be 2*order + 1? seems fine for piecewise, but for higher orders?
        raise ValueError("nbins must be at least twice the order")

    kvs = _get_knot_vectors(rmin, rmax, nbins, order)
    rcont = np.linspace(rmin, rmax, ncont)
    bases = np.empty((nbins+1, ncont))
    bases[0,:] = rcont
    for n in range(nbins):
        kv = kvs[n]
        b = BSpline.basis_element(kv)
        bases[n+1,:] = [b(r) if kv[0]<=r<=kv[-1] else 0 for r in rcont]

    np.savetxt(projfn, bases.T)
    return projfn


def _get_knot_vectors(rmin, rmax, nbins, order):
    nknots = order+2
    kvs = np.empty((nbins, nknots))
    width = (rmax-rmin)/(nbins-order)
    for i in range(order):
        val = i+1
        kvs[i,:] = np.concatenate((np.full(nknots-val, rmin), np.linspace(rmin+width, rmin+width*val, val)))
        kvs[nbins-i-1] = np.concatenate((np.linspace(rmax-width*val, rmax-width, val), np.full(nknots-val, rmax)))
    for j in range(nbins-2*order):
        idx = j+order
        kvs[idx] = rmin+width*j + np.arange(0,nknots)*width                     
    return kvs


#############
# BAO basis #
#############

def bao_bases(rmin, rmax, projfn, cosmo_base, ncont=1000, redshift=0, 
                dalpha=0.01, alpha_model=1.0, bias=1.0, k0=0.1):
    if not cosmo_base:
        raise ValueError("Must pass cosmo_base!")

    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)

    def cf_model(s):
        return bias * CF(s)

    rcont = np.linspace(rmin, rmax, ncont)
    bs = _get_bao_components(rcont, cf_model, dalpha, alpha_model, k0=k0)

    nbases = len(bs)    
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont
    bases[1:nbases+1,:] = bs
    
    np.savetxt(projfn, bases.T)
    nprojbins = bases.shape[0]-1
    return nprojbins, projfn


def _get_bao_components(s, cf_func, dalpha, alpha, k0=0.1):   
    print("updated bases!!")
    k1 = 10.0
    b1 = k1/s**2
    
    k2 = 0.1
    b2 = k2/s

    k3 = 0.001
    b3 = k3*np.ones(len(s))
    
    cf = cf_func(alpha*s)
    b4 = cf

    cf_dalpha = cf_func((alpha+dalpha)*s)
    dcf_dalpha = _partial_derivative(cf, cf_dalpha, dalpha)
    b5 = k0*dcf_dalpha
    
    return b1,b2,b3,b4,b5
    

def _partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv    

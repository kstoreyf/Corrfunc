import numpy as np
from nbodykit.lab import cosmology


def write_bases(rmin, rmax, saveto, ncont=1000, **kwargs):
    bases = get_bases(rmin, rmax, ncont=ncont, **kwargs)
    np.savetxt(saveto, bases.T)
    nprojbins = bases.shape[0]-1
    return nprojbins, saveto


def bao_bases(s, cf_func, p1=1, p2=1, p3=1):
    
    b1 = p1 * 1.0/s**2
    b2 = p2 * 1.0/s
    b3 = p3 * np.ones(len(s))
    
    cf = cf_func(s)
    b4 = cf

    alpha = 1.001
    dalpha = 1-alpha
    dxi_dalpha = partial_derivative(cf, cf_func(alpha*s), dalpha)
    b5 = dalpha*dxi_dalpha
    
    return b1,b2,b3,b4,b5


def get_bases(rmin, rmax, ncont=1000, cosmo_base=None, redshift=0):

    if not cosmo_base:
        raise ValueError("Must pass cosmo_base!")

    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)

    def cf_model(s):
        alpha_model = 1.01
        return CF(alpha_model*s)

    rcont = np.linspace(rmin, rmax, ncont)
    #bs = bao_bases(rcont, CF)
    bs = bao_bases(rcont, cf_model)

    nbases = len(bs)    
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont
    bases[1:nbases+1,:] = bs

    return bases
    

def partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv    

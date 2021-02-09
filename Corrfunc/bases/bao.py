import numpy as np
from nbodykit.lab import cosmology

'''
Helper routines for BAO basis functions
'''

def write_bases(rmin, rmax, saveto, ncont=1000, **kwargs):
    bases = get_bases(rmin, rmax, ncont=ncont, **kwargs)
    np.savetxt(saveto, bases.T)
    nprojbins = bases.shape[0]-1
    return nprojbins, saveto


def bao_bases(s, cf_func, dalpha, alpha, k0=0.1):   
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
    dcf_dalpha = partial_derivative(cf, cf_dalpha, dalpha)
    b5 = k0*dcf_dalpha
    
    return b1,b2,b3,b4,b5


def get_bases(rmin, rmax, ncont=1000, cosmo_base=None, redshift=0, dalpha=0.01, alpha_model=1.0, bias=1.0, k0=0.1):

    if not cosmo_base:
        raise ValueError("Must pass cosmo_base!")

    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)

    #dalpha = 0.01
    #alpha_model = 1.02
    print("bias: {}. dalpha: {}, alpha_model: {}".format(bias, dalpha, alpha_model))

    #def cf_model(s, alpha_model):
        #return bias * CF(alpha_model*s)
    def cf_model(s):
        return bias * CF(s)

    rcont = np.linspace(rmin, rmax, ncont)
    #bs = bao_bases(rcont, CF)
    bs = bao_bases(rcont, cf_model, dalpha, alpha_model, k0=k0)

    nbases = len(bs)    
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont
    bases[1:nbases+1,:] = bs

    return bases
    

def partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv    

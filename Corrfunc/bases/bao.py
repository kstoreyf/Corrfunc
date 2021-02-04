import numpy as np
from nbodykit.lab import cosmology

'''
.. module:: suave

Helper routines for BAO basis functions
'''

def write_bases(rmin, rmax, saveto, ncont=1000, **kwargs):
    bases = get_bases(rmin, rmax, ncont=ncont, **kwargs)
    np.savetxt(saveto, bases.T)
    nprojbins = bases.shape[0]-1
    return nprojbins, saveto


def bao_bases(s, cf_func, dalpha, alpha_model):
    
    b1 = 1.0/s**2
    b2 = 0.1/s
    b3 = 0.001*np.ones(len(s))
    
    cf = cf_func(s, alpha_model=alpha_model)
    b4 = cf

    alpha = dalpha/alpha_model + 13
    #dalpha = alpha_model*alpha - alpha_model
    cf_dalpha = cf_func(alpha*s, alpha_model=alpha_model)
    dcf_dalpha = partial_derivative(cf, cf_dalpha, dalpha)
    b5 = dalpha*dcf_dalpha
    
    return b1,b2,b3,b4,b5


def get_bases(rmin, rmax, ncont=1000, cosmo_base=None, redshift=0, dalpha=0.1, alpha_model=1.0, bias=1.0):

    if not cosmo_base:
        raise ValueError("Must pass cosmo_base!")

    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)

    #dalpha = 0.01
    #alpha_model = 1.02
    print("bias: {}. dalpha: {}, alpha_model: {}".format(bias, dalpha, alpha_model))

    def cf_model(s, alpha_model):
        return bias * CF(alpha_model*s)

    rcont = np.linspace(rmin, rmax, ncont)
    #bs = bao_bases(rcont, CF)
    bs = bao_bases(rcont, cf_model, dalpha, alpha_model)

    nbases = len(bs)    
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont
    bases[1:nbases+1,:] = bs

    return bases
    

def partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv    

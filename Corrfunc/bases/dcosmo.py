import numpy as np
from nbodykit.lab import cosmology


def write_bases(rmin, rmax, saveto, ncont=300, **kwargs):
    bases = get_bases(rmin, rmax, ncont=ncont, **kwargs)
    np.savetxt(saveto, bases.T)
    nprojbins = bases.shape[0]-1
    return nprojbins, saveto


def get_bases(rmin, rmax, ncont=300, params=None, cosmo_base=None, redshift=0):
    if params is None or cosmo_base is None:
        raise ValueError("Must pass params and cosmo_base!")
    
    nbases = len(params)+1
    rcont = np.linspace(rmin, rmax, ncont)
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont

    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)
    xi_base = CF(rcont)
    bases[1,:] = xi_base

    for i in range(len(params)):
        param = params[i]
        cosmo_dict = dict(cosmo_base)
        val_base = cosmo_dict[param]
        dval = val_base * 0.01
        val_new = val_base + dval
        cosmo_dict[param] = val_new
        cosmo = cosmology.Cosmology.from_dict(cosmo_dict)
        
        Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
        CF = cosmology.correlation.CorrelationFunction(Plin)
        xi = CF(rcont)
    
        dcosmo = partial_derivative(xi_base, xi, dval)
        bases[i+2, :] = dcosmo*dval

    return bases
    

def partial_derivative(f1, f2, dv):
    df = f2-f1
    deriv = df/dv
    return deriv 

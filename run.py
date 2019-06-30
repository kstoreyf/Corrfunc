import numpy as np
import pandas as pd
from scipy import interpolate
from astropy.cosmology import LambdaCDM
import Corrfunc
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from astropy.cosmology import LambdaCDM
import time

import plotter


def main():

    #nd = 1012
    nd = 10
    data1fn = '../lss/mangler/samples/a0.6452_0001.v5_ngc_ifield_ndata{}.rdzw'.format(nd)
    rand1fn = '../lss/mangler/samples/a0.6452_rand20x.dr12d_cmass_ngc_ifield_ndata{}.rdz'.format(nd)
    data2fn = data1fn
    rand2fn = rand1fn

    print 'Running for n_data={}'.format(nd)

    K = 20
    pimax = 40 #Mpc/h
    rpmin = 0.5
    rpmax = 80 #Mpc/h

    #rpbins = np.array([0.1, 1., 10.])
    rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), K+1)
    rpbins_avg = 0.5 * (rpbins[1:] + rpbins[:-1])

    start = time.time()
    wp, wprp_corrfunc, wprp_nopi = run_corrfunc(data1fn, rand1fn, data2fn, rand2fn, rpbins, pimax)
    end = time.time()
    print 'Time: {:3f} s'.format(end-start)

    # rps = [rpbins_avg, rpbins_avg, rpbins_avg]
    # wprps = [wp, wprp_corrfunc, wprp_nopi]
    # labels = ['wp built-in', 'wp calculated', 'wp no pi']
    rps = [rpbins_avg]
    wprps = [wp]
    labels = ['wp built-in']
    #plotter.plot_wprp(rps, wprps, labels, wp_tocompare='wp built-in')




def run_corrfunc(data1fn, rand1fn, data2fn, rand2fn, rpbins, pimax):
    print 'Loading data'
    data1 = pd.read_csv(data1fn)
    rand1 = pd.read_csv(rand1fn)
    data2 = pd.read_csv(data2fn)
    rand2 = pd.read_csv(rand2fn)

    #can only do autocorrelations right now
    wp, wprp_corrfunc, wprp_nopi = counts(data1['ra'].values, data1['dec'].values, data1['z'].values,
                    rand1['ra'].values, rand1['dec'].values, rand1['z'].values,
                    rpbins, pimax, comoving=True)

    return wp, wprp_corrfunc, wprp_nopi



cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

def counts(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, rpbins, pimax,
         weights_data=None, weights_rand=None, pibinwidth=1, comoving=False):

    assert(len(ra_data)==len(dec_data) and len(ra_data)==len(z_data))
    assert(len(ra_rand)==len(dec_rand) and len(ra_rand)==len(z_rand))

    ndata = len(ra_data)
    nrand = len(ra_rand)
    nbins = len(rpbins)-1
    pibins = np.arange(0, pimax + pibinwidth, pibinwidth)

    if comoving:
        zdf = pd.DataFrame(z_data)
        z_data = zdf.apply(get_comoving_dist)[0].values
        rzdf = pd.DataFrame(z_rand)
        z_rand = rzdf.apply(get_comoving_dist)[0].values

    dd_res_corrfunc = DDrppi_mocks(1, 2, 0, pimax, rpbins, ra_data, dec_data, z_data, is_comoving_dist=comoving)
    dr_res_corrfunc = DDrppi_mocks(0, 2, 0, pimax, rpbins, ra_data, dec_data, z_data,
                                        RA2=ra_rand, DEC2=dec_rand, CZ2=z_rand, is_comoving_dist=comoving)
    rr_res_corrfunc = DDrppi_mocks(1, 2, 0, pimax, rpbins, ra_rand, dec_rand, z_rand, is_comoving_dist=comoving)

    wp = convert_rp_pi_counts_to_wp(ndata, ndata, nrand, nrand, dd_res_corrfunc, dr_res_corrfunc,
                                    dr_res_corrfunc, rr_res_corrfunc, nbins, pimax)

    dd_rp_pi_corrfunc = np.zeros((len(pibins) - 1, len(rpbins) - 1))
    dr_rp_pi_corrfunc = np.zeros((len(pibins) - 1, len(rpbins) - 1))
    rr_rp_pi_corrfunc = np.zeros((len(pibins) - 1, len(rpbins) - 1))

    for m in range(len(pibins)-1):
        for n in range(len(rpbins)-1):
            idx = (len(pibins)-1) * n + m
            dd_rp_pi_corrfunc[m][n] = dd_res_corrfunc[idx][4]
            dr_rp_pi_corrfunc[m][n] = dr_res_corrfunc[idx][4]
            rr_rp_pi_corrfunc[m][n] = rr_res_corrfunc[idx][4]

    estimator_corrfunc = calc_ls(dd_rp_pi_corrfunc, dr_rp_pi_corrfunc, rr_rp_pi_corrfunc, ndata, nrand)
    wprp_corrfunc = 2*pibinwidth*np.sum(estimator_corrfunc, axis=0)
    est_ls, wprp_nopi = calc_wprp_nopi(dd_rp_pi_corrfunc, dr_rp_pi_corrfunc, rr_rp_pi_corrfunc, ndata, nrand)

    return wp, wprp_corrfunc, wprp_nopi


def calc_wprp_nopi(dd, dr, rr, ndata, nrand):
    dd = np.sum(dd, axis=0)
    dr = np.sum(dr, axis=0)
    rr = np.sum(rr, axis=0)

    est_ls = calc_ls(dd, dr, rr, ndata, nrand)
    #wprp = 2*np.sum(est_ls, axis=0)
    wprp = 2*est_ls

    return est_ls, wprp


def get_comoving_dist(z):
    comov = cosmo.comoving_distance(z)
    return comov.value*cosmo.h

def calc_ls(dd_counts, dr_counts, rr_counts, ndata, nrand):
    fN = float(nrand)/float(ndata)
    return (fN*fN*dd_counts - 2*fN*dr_counts + rr_counts)/rr_counts


if __name__=='__main__':
    main()
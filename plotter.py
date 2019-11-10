from matplotlib import pyplot as plt
import numpy as np





def plot_wprp(rps, wprps, labels, wp_tocompare=None):

    if wp_tocompare:
        compidx = labels.index(wp_tocompare)

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    for i in range(len(labels)):
        rp = rps[i]
        wprp = np.array(wprps[i])
        label = labels[i]

        ax0.loglog(rp, wprp, label=label, marker='o', ls="None", color='grey')

        plt.xlabel(r'$r_p$ (Mpc/h)')
        ax0.set_ylabel(r'$w_p$($r_p$)')
        ax1.set_ylabel(r'$w_p$/$w_{{p,\mathrm{{{0}}}}}$'.format(wp_tocompare))

        ax0.legend(loc='best')

        if wp_tocompare:
            wpcomp = wprps[compidx]

            if len(wprp)==len(wpcomp):
                ax1.semilogx(rp, wprp/wpcomp, color='grey')

    plt.show()
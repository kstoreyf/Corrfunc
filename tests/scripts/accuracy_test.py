import numpy as np 
from Corrfunc.io import read_catalog
from Corrfunc.theory.DDsmu import DDsmu


def main():

    savetag = ''
    proj_type = 'tophat'
    nprojbins = 9
    r_edges = np.linspace(10., 100., nprojbins+1)
    proj_dict = {'tophat': {'nprojbins': nprojbins,
                            'proj_func': tophat,
                            'proj_fn': None,
                            'args':[r_edges],
                            'kwargs':{}
                            }}

    proj = proj_dict[proj_type]
    frac = 0.0001
    allx, ally, allz = read_catalog()
    N = np.int(frac * len(allx))
    x = np.random.choice(allx, N, replace=False)
    y = np.random.choice(ally, N, replace=False)
    z = np.random.choice(allz, N, replace=False)
    data = np.array([x,y,z]).T
    fmt='%10.10f'

    ### Brute force test
    v_dd_correct, T_dd_correct = dd_bruteforce(data, proj['proj_func'], proj['nprojbins'], *proj['args'], **proj['kwargs'])

    np.save(f'../output/correct_full_{proj_type}.npy', [v_dd_correct, T_dd_correct, proj_type, proj])
    np.savetxt(f'../output/correct_vdd_{proj_type}.npy', v_dd_correct, fmt=fmt)
    np.savetxt(f'../output/correct_Tdd_{proj_type}.npy', T_dd_correct, fmt=fmt)
    print(v_dd_correct)
    print(T_dd_correct)

    ### Corrfunc/suave test
    nthreads = 1
    mumax = 1.0
    nmubins = 1
    _, v_dd, T_dd = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=proj['nprojbins'], projfn=proj['proj_fn'], periodic=False)
    T_dd = T_dd.reshape((nprojbins, nprojbins)) #make code output it like this?! or maybe i didn't because it makes it easier to pass directly to compute_amps, etc
    print(v_dd)
    print(T_dd)

    np.save(f'../output/suave_full_{proj_type}.npy', [v_dd, T_dd, proj_type, proj])
    np.savetxt(f'../output/suave_vdd_{proj_type}.npy', v_dd, fmt=fmt)
    np.savetxt(f'../output/suave_Tdd_{proj_type}.npy', T_dd, fmt=fmt)


def dd_bruteforce(data, proj_func, nprojbins, *args, **kwargs):
    # data is shape (N, 3)
    N = data.shape[0]
    v_dd = np.zeros(nprojbins)
    T_dd = np.zeros((nprojbins, nprojbins))
    for i in range(N): 
        for j in range(N):
            if i!=j:
                # this is the euclidean distance (https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy)
                r = np.linalg.norm(data[i]-data[j])
                u = proj_func(r, *args, **kwargs)
                v_dd += u
                T_dd += np.outer(u, u)
    return v_dd, T_dd


def tophat(r, r_edges):
    u = np.empty(len(r_edges)-1)
    for bb in range(len(r_edges)-1):
        if (r_edges[bb] < r < r_edges[bb+1]):
            u[bb] = 1.0
        else:
            u[bb] = 0.0
    return u
        


if __name__=='__main__':
    main()
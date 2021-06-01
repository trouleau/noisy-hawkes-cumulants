import multiprocessing
import itertools
import numpy as np
import pandas as pd
import scipy.optimize
import pickle

import sys
if '../' not in sys.path:
    sys.path.append('../')

import tick
from tick.hawkes.simulation import SimuHawkesExpKernels
from tick.hawkes.inference import HawkesConditionalLaw, HawkesADM4, HawkesCumulantMatching, HawkesSumGaussians

from desync_mhp.lib.inference import HawkesExpKernConditionalMLE

import lib

G_true = np.array([[0.23, 0.23, 0.23, 0.23, 0.23, 0.  , 0.  , 0.  , 0.  , 0.23],
                   [0.  , 0.23, 0.23, 0.23, 0.23, 0.  , 0.  , 0.  , 0.23, 0.  ],
                   [0.  , 0.  , 0.23, 0.23, 0.23, 0.  , 0.  , 0.  , 0.  , 0.  ],
                   [0.  , 0.  , 0.  , 0.23, 0.23, 0.  , 0.  , 0.  , 0.  , 0.  ],
                   [0.  , 0.  , 0.  , 0.  , 0.23, 0.  , 0.  , 0.  , 0.  , 0.  ],
                   [0.  , 0.  , 0.  , 0.  , 0.  , 0.23, 0.  , 0.  , 0.  , 0.  ],
                   [0.  , 0.23, 0.  , 0.  , 0.  , 0.23, 0.23, 0.  , 0.  , 0.  ],
                   [0.23, 0.  , 0.  , 0.  , 0.  , 0.23, 0.23, 0.23, 0.  , 0.  ],
                   [0.  , 0.  , 0.  , 0.  , 0.  , 0.23, 0.23, 0.23, 0.23, 0.  ],
                   [0.  , 0.  , 0.  , 0.  , 0.  , 0.23, 0.23, 0.23, 0.23, 0.23]])
dim = len(G_true)
decay = 1.0
mu_true = 0.01 * np.ones(dim)
# Compute the ground-truth cumulants
L_true, C_true, Kc_true = lib.utils.cumulants.compute_cumulants(G=G_true, mus=mu_true,)


def simulate(noise_scale, seed):
    # Sample noise
    noise_rand_state = np.random.RandomState(seed=None)
    noise_dist_arr =  ['exponential' for _ in range(dim)]
    noise_scale_arr = [noise_scale for _ in range(dim)]
    # Init mhp simulation object
    simu_hawkes = SimuHawkesExpKernels(adjacency=G_true, decays=decay,
                                       baseline=mu_true, max_jumps=0,
                                       verbose=False)
    # Build noisy simulation object
    simu_noisy_hawkes = lib.simulation.noisy_hawkes.SimulatorNoisyHawkesCustomKernels(
        simu_obj=simu_hawkes,
        noise_dist=noise_dist_arr,
        noise_scale=noise_scale_arr,
        burn_in_quantile=0.99,
        num_real=5,
        num_jumps=100000,
        seed=seed,
        no_multi=True)
    # Simulate noisy data
    noisy_events = simu_noisy_hawkes.simulate()
    return noisy_events


def find_best_integration_support(events, max_iter=20, initial_simplex=[[10.0], [50.0]], verbose=False):
    def int_support_loss(H, events):
        nphc = HawkesCumulantMatching(integration_support=float(H), max_iter=0, verbose=False)
        nphc.fit(events)
        skew_loss = np.linalg.norm(nphc.skewness - Kc_true, ord=2)
        cov_loss = np.linalg.norm(nphc.covariance - C_true, ord=2)
        if verbose:
            print(f"{float(H):>6.2f}, loss={loss:.2e}, skew_loss={skew_loss:.2e}, cov_loss={cov_loss:.2e}")
        return skew_loss
    res = scipy.optimize.minimize(
        int_support_loss,
        x0=20.0,
        args=(events,),
        options={'max_iter': max_iter,
                 'maxfev': max_iter,
                 'initial_simplex': initial_simplex},
        method='Nelder-Mead')
    return float(res.x)


if __name__ == "__main__":

    # Define experiments
    noise_scale_range = np.logspace(-1, 1.4, 25)
    sim_seed_list = np.random.RandomState(703994370).randint(0, 2**32 - 1, size=20)
    args_iter = list(itertools.product(noise_scale_range, sim_seed_list))

    data = list()
    for it, (noise_scale, sim_seed) in enumerate(args_iter):
        print()
        print(f"Iter {it:>2d}/{len(args_iter):>2d} | noise_scale: {noise_scale:.2e}...")

        # Simulate data
        noisy_events = simulate(noise_scale=noise_scale, seed=sim_seed)

        # ADM4
        adm4 = HawkesADM4(decay=1.0, verbose=False)
        adm4.fit(noisy_events)
        print(f"ADM4: done.")

        # NPHC
        H = find_best_integration_support(noisy_events)
        nphc = HawkesCumulantMatching(integration_support=H, max_iter=20000, verbose=False)
        nphc.fit(noisy_events)
        print(f"NPHC: done.")

        # WH
        wh = HawkesConditionalLaw(delta_lag=0.1, min_lag=0.0001, max_lag=100.0,
                                  n_quad=20, max_support=10.0)
        wh.fit(noisy_events)
        wh_adj = wh.get_kernel_norms()
        print(f"WH: done.")

        # Desync-MLE
        end_time = max([max(map(max, real)) for real in noisy_events])
        # Desync-MLE
        desyncmle = HawkesExpKernConditionalMLE(
            decay=1.0,
            noise_penalty='l2', noise_C=1e3,
            hawkes_penalty='l1', hawkes_base_C=1e2, hawkes_adj_C=1e5,
            solver='sgd', tol=1e-4, max_iter=1000,
            verbose=False
        )
        desyncmle.fit(noisy_events, end_time=end_time,
                    z_start=np.zeros(dim),
                    theta_start=np.hstack((
                        0.01*np.ones(dim),
                        np.random.uniform(0.0, 0.1, size=dim**2)
                    )),
                    callback=None)
        desyncmle_adj = np.reshape(desyncmle.coeffs[2*dim:], (dim, dim))
        print(f"Desync-MLE: done")

        # Store results
        data.append({
            'noise_scale': noise_scale,
            'sim_seed': sim_seed,
            'adm4_adj': adm4.adjacency.copy(),
            'nphc_adj': nphc.adjacency.copy(),
            'wh_adj': wh_adj,
            'desyncmle_adj': desyncmle_adj,
            'mean': nphc.mean_intensity.copy(),
            'cov': nphc.covariance.copy(),
            'skew': nphc.skewness.copy(),
            'H': H,
        })

    # Save the results
    pd.DataFrame(data).to_pickle('res-synthetic.pkl')

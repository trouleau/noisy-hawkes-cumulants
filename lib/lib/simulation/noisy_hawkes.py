from multiprocessing import cpu_count
import sys
import copy
import pickle
import numpy as np
import scipy.stats

import tick.hawkes


def num_jumps_to_end_time(baselines, adjacency, decays, num_jumps):
    """
    Given Hawkes process parameters, compute the size `end_time` of observation
    window necessary to simulate `num_jumps` events in expectation.
    """
    simu_hawkes = tick.hawkes.SimuHawkesExpKernels(
        baseline=baselines, adjacency=adjacency, decays=decays,
        max_jumps=num_jumps, seed=None, verbose=False)
    tot_mean_intensity = simu_hawkes.mean_intensity().sum()
    end_time = num_jumps / simu_hawkes.mean_intensity().sum()
    return end_time


def end_time_to_num_jumps(baselines, adjacency, decays, end_time, **kwargs):
    """
    Given Hawkes process parameters, compute the size `end_time` of observation
    window necessary to simulate `num_jumps` events in expectation.
    """
    simu_hawkes = tick.hawkes.SimuHawkesExpKernels(
        baseline=baselines, adjacency=adjacency,  decays=decays,
        end_time=end_time, seed=None, verbose=False)
    tot_mean_intensity = simu_hawkes.mean_intensity().sum()
    num_jumps = end_time * simu_hawkes.mean_intensity().sum()
    return num_jumps


def num_jumps_to_end_time(simu_obj, num_jumps):
    """
    Given Hawkes process object, compute the size `end_time` of observation
    window necessary to simulate `num_jumps` events in expectation.
    """
    mean_int_arr = simu_obj.mean_intensity()
    if mean_int_arr.min() < 0:
        raise RuntimeError("Process is unstable. Operation not permitted.")
    end_time = num_jumps / mean_int_arr.sum()
    return end_time


def num_jumps_to_end_time(simu_obj, num_jumps):
    """
    Given Hawkes process object, compute the size `end_time` of observation
    window necessary to simulate `num_jumps` events in expectation.
    """
    mean_int_arr = simu_obj.mean_intensity()
    if mean_int_arr.min() < 0:
        raise RuntimeError("Process is unstable. Operation not permitted.")
    end_time = num_jumps / mean_int_arr.sum()
    return end_time


def simulate_exp_hawkes_with_translation_noise(*args, return_original_events=False, return_sim_obj=False, **kwargs):
    sim_obj = SimulatorNoisyHawkesExpKernels(*args, **kwargs)
    trans_events, all_events = sim_obj.simulate(return_original_events=True)
    if return_sim_obj:
        return trans_events, all_events, sim_obj
    elif return_original_events:
        return trans_events, all_events
    else:
        return trans_events


class SimulatorNoisyHawkes:

    def __init__(self, *, noise_dist, noise_scale, num_real, end_time=None, num_jumps=None,
                 seed=None, burn_in_quantile=0.95, no_multi=False):
        assert hasattr(self, 'simu_obj'), "`simu_obj` attribute is not set... Probably a bug in the child class init..."

        # Simulation parameters
        self.num_real = num_real  # Number of realizations
        if not ((end_time is None) ^ (num_jumps is None)):
            print(end_time, num_jumps)
            raise RuntimeError("Exactly one of `end_time` or `num_jumps` should be provided")
        if end_time is None:
            end_time = num_jumps_to_end_time(self.simu_obj, num_jumps=num_jumps)
        self.end_time = end_time  # Length of observation window
        self.num_jumps = num_jumps  # Number of observations

        if seed is None:  # If no random seed is provided, sample one
            print('WARNING: Set random seed')
            np.random.seed(None)
            seed = np.random.randint(low=0, high=2**31 - 1)
        self.seed = seed  # Random seed of simulation
        self.rand_state = np.random.RandomState(self.seed)

        self.noise_dist = noise_dist  # Noise distribution
        self.noise_scale = noise_scale  # Noise scale
        self.burn_in_quantile = burn_in_quantile  # Quantile to use for burn-in

        # Enforce single-threaded for simulation
        self.no_multi = no_multi

    def to_pickle(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)


    def _prepare_simulation(self):
        """
        Build the functions to sample the from the noise distribution for each
        realization and dimensions, and compute the global values of burn-in
        periods to add at the begining and end of the simulation period.

        Set the following attributes:
        -----------------------------
        noise_func_dict : dict of callable
            The function to sample from the noise distribution. It takes as
            input parameters `(ev_i)` where `ev_i` is the array of simulated
            events (necessary to know how many values to samples)
        burn_in_start : float
            Burn-in necessary at the start of the simulation for this noise
        burn_in_end : float
            Burn-in necessary at the end of the simulation for this noise
        z_true_dict : dict of np.ndarray
            Hold the ground-truth noise values (only for the synchronized noises)
        """

        dim  = len(self.noise_dist)

        # Set random seed for noise sampling
        nose_random_state = np.random.RandomState(self.rand_state.randint(low=0, high=2**31 - 1))

        # Global values burn-in periods (shared among all real and dimensions)
        self.burn_in_start = 0.0
        self.burn_in_end = 0.0
        # Dict of noise distribution for each real and dimension
        self.noise_func_dict = {r: {} for r in range(self.num_real)}

        self.z_true_dict = {}  # dict of sync-noise values

        for i in range(dim):

            if self.noise_dist[i] == 'exponential':
                for r in range(self.num_real):
                    # Burn-in
                    b_start = -max(self.noise_scale) * np.log(1 - self.burn_in_quantile)
                    b_end = 0.0
                    self.burn_in_start = max(self.burn_in_start, b_start)
                    self.burn_in_end = max(self.burn_in_end, b_end)
                    # Noise func
                    def noise_func(real_idx, dim_idx, ev_i):
                        return nose_random_state.exponential(scale=self.noise_scale[dim_idx], size=len(ev_i))
                    self.noise_func_dict[r][i] = noise_func

            elif self.noise_dist[i] == 'gaussian':
                for r in range(self.num_real):
                    # Burn-in
                    ppf_val = scipy.stats.norm.ppf(1 - (1 - self.burn_in_quantile) / 2)
                    b_start = ppf_val * max(self.noise_scale)
                    b_end = ppf_val * max(self.noise_scale)
                    self.burn_in_start = max(self.burn_in_start, b_start)
                    self.burn_in_end = max(self.burn_in_end, b_end)
                    # Noise func
                    def noise_func(real_idx, dim_idx, ev_i):
                        return nose_random_state.normal(scale=self.noise_scale[dim_idx], size=len(ev_i))
                    self.noise_func_dict[r][i] = noise_func

            elif self.noise_dist[i] == 'sync-exponential':
                # Sample all noise values
                self.z_true_dict[i] = nose_random_state.exponential(
                    scale=self.noise_scale[i], size=self.num_real)
                for r in range(self.num_real):
                    # Burn-in
                    b_start =  abs(self.z_true_dict[i].max())
                    b_end = abs(self.z_true_dict[i].min())
                    self.burn_in_start = max(self.burn_in_start, b_start)
                    self.burn_in_end = max(self.burn_in_end, b_end)
                    # Noise func
                    def noise_func(real_idx, dim_idx, ev_i):
                        return self.z_true_dict[dim_idx][real_idx] * np.ones_like(ev_i)
                    self.noise_func_dict[r][i] = noise_func

            elif self.noise_dist[i] == 'sync-gaussian':
                # Sample all noise values
                self.z_true_dict[i] = nose_random_state.normal(
                    scale=self.noise_scale[i], size=self.num_real)
                for r in range(self.num_real):
                    # Burn-in
                    b_start = abs(self.z_true_dict[i].max())
                    b_end = abs(self.z_true_dict[i].min())
                    self.burn_in_start = max(self.burn_in_start, b_start)
                    self.burn_in_end = max(self.burn_in_end, b_end)
                    # Noise func
                    def noise_func(real_idx, dim_idx, ev_i):
                        return self.z_true_dict[dim_idx][real_idx] * np.ones_like(ev_i)
                    self.noise_func_dict[r][i] = noise_func

            else:
                raise ValueError('Invalid noise distribution')

        # Compute the value of the extended end time with burn-ins
        self.extended_end_time = (self.burn_in_start + self.end_time
                                  + self.burn_in_end)

    def _post_simulation(self):
        self.noise_func_dict = None  # Remove local functions to make picklable

    def _add_noise(self, all_events):
        trans_events = list()
        for r, events_r in enumerate(all_events):  # For each realization
            trans_events_r = list()
            for i, ev_i in enumerate(events_r):  # For each dimension
                # Sample random translations from the noise distribution
                noise_i = self.noise_func_dict[r][i](r, i, ev_i)
                # print(f"r={r} i={i} event:", ev_i)
                # print(f"r={r} i={i} noise:", noise_i)
                # Translate events
                trans_ev_i = np.sort(ev_i + noise_i)
                # print(f"r={r} i={i} trans before:", trans_ev_i)
                # Remove burn-in periods
                trans_ev_i -= self.burn_in_start # Remove burn-in start time
                # print(f"r={r} i={i} trans mid:   ", trans_ev_i)
                trans_ev_i = trans_ev_i[
                    (trans_ev_i >= 0) &      # Filter-out burn-in start period
                    (trans_ev_i < self.end_time)  # Filter-out burn-in end period
                ]
                trans_events_r.append(trans_ev_i)
                # print(f"r={r} i={i} trans after: ", trans_ev_i)
                # print("end_time =", self.end_time)
            trans_events.append(trans_events_r)
        return trans_events

    def _remove_burn_in_periods(self, all_events):
        filtered_events = list()
        for r, events_r in enumerate(all_events):  # For each realization
            filtered_events_r = list()
            for i, ev_i in enumerate(events_r):  # For each dimension
                # Remove burn-in periods
                ev_i -= self.burn_in_start # Remove burn-in start time
                ev_i = ev_i[
                    (ev_i >= 0) &      # Filter-out burn-in start period
                    (ev_i < self.end_time)  # Filter-out burn-in end period
                ]
                filtered_events_r.append(ev_i)
            filtered_events.append(filtered_events_r)
        return filtered_events

    def _simulate_noise_free(self):
         # Sample list of seeds for all realizations
        seed_list = self.rand_state.randint(low=0, high=2**31 - 1, size=self.num_real)
        if not self.no_multi:
            try:
                return self._simulate_noise_free_multi(seed_list)
            except AssertionError:  # In case we were already in a deamon process
                print('WARNING: Switching to single-threaded simulation...', file=sys.stderr, flush=True)
        # Simulate single-threaded
        return self._simulate_noise_free_single_thread(seed_list)

    def _simulate_noise_free_multi(self, seed_list):
        self.simu_obj.reset()  # NOTE: Must reset first to make several simulate call work as expected (limitation of `tick` simulation objects)
        multi = tick.hawkes.SimuHawkesMulti(self.simu_obj,
                                            n_simulations=self.num_real,
                                            n_threads=max(self.num_real,
                                                          cpu_count()-1))
        # Set seed and end time for each realization
        for i, seed in enumerate(seed_list):
            multi._simulations[i].seed = int(seed)  # Set new random seed
            multi._simulations[i].end_time = self.extended_end_time  # Set end time with burn-in periods
        multi.simulate()
        return multi.timestamps

    def _simulate_noise_free_single_thread(self, seed_list):
        # Simulate realizations in sequence
        all_events = list()
        for seed in seed_list:
            self.simu_obj.reset()  # Reset object for a new simulation
            self.simu_obj.seed = int(seed)  # Set new random seed
            self.simu_obj.end_time = self.extended_end_time  # Set end time with burn-in periods
            self.simu_obj.simulate()   # Simulate realization
            all_events.append(self.simu_obj.timestamps)
        return all_events

    def simulate(self, return_original_events=False):
        # Build the noise functions and burn-in periods
        self._prepare_simulation()
        # Simulate the process (without noise) with extended observation window
        all_events = self._simulate_noise_free()
        # Add the translation noise
        trans_events = self._add_noise(all_events)
        self.trans_events = trans_events
        # Post-process (to make picklable)
        self._post_simulation()
        # Return
        if return_original_events:
            # Remove burn-in period of 'noise-free' events
            all_events = self._remove_burn_in_periods(all_events)
            self.orig_events = all_events
            return trans_events, all_events
        return trans_events


class SimulatorNoisyHawkesExpKernels(SimulatorNoisyHawkes):
    """
    Simulate Multivariate Hawkes Processes with translation noise.
    """

    def __init__(self, *, baselines, adjacency, decays, noise_dist, noise_scale,
                 num_real, end_time=None, num_jumps=None, seed=None,
                 burn_in_quantile=0.95, no_multi=False):
        # Model parameters
        self.baselines = baselines
        self.adjacency = adjacency
        self.decays = decays
        self.dim = len(baselines)
        self.simu_obj = tick.hawkes.SimuHawkesExpKernels(
            adjacency=adjacency, decays=decays, baseline=baselines,
            seed=None, verbose=False)
        # NOTE: `end_time` is set on simulation time (after noise distribution
        # and burn-in periods are set)

        # Init parent
        super().__init__(noise_dist=noise_dist, noise_scale=noise_scale,
                         num_real=num_real, end_time=end_time,
                         num_jumps=num_jumps, seed=seed,
                         burn_in_quantile=burn_in_quantile, no_multi=no_multi)


class SimulatorNoisyHawkesCustomKernels(SimulatorNoisyHawkes):
    """
    Simulate Multivariate Hawkes Processes with translation noise.
    """

    def __init__(self, *, simu_obj, noise_dist, noise_scale,
                 num_real, end_time=None, num_jumps=None, seed=None,
                 burn_in_quantile=0.95, no_multi=False):
        # Set simulation obj
        if isinstance(simu_obj, tick.hawkes.SimuHawkes):
            self.simu_obj = copy.deepcopy(simu_obj)
            self.simu_obj.reset()
            self.simu_obj.end_time = None
            self.simu_obj.max_jumps = None
            self.baselines = simu_obj.baseline
        # NOTE: `end_time` is set on simulation time (after noise distribution
        # and burn-in periods are set)

        # Init parent
        super().__init__(noise_dist=noise_dist, noise_scale=noise_scale,
                         num_real=num_real, end_time=end_time,
                         num_jumps=num_jumps, seed=seed,
                         burn_in_quantile=burn_in_quantile, no_multi=no_multi)

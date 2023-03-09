import numpy as np
from numba import jit


@jit(nopython=True)
def find_loc_peaks(slope, energy):
    num_crit = len(energy)

    loc_peaks = np.zeros_like(slope)

    for idx, val in enumerate(slope):  # ii in range(len(slope)):
        n = idx
        if val > 0:
            while (n < num_crit - 1) and (slope[n] > 0):
                n = n + 1
            loc_peaks[idx] = energy[n - 1]
        else:
            while (n >= 0) and (slope[n] <= 0):
                n = n - 1
            loc_peaks[idx] = energy[n + 1]

    return loc_peaks

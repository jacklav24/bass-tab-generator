# THIS FILE IS A PYTHON CONVERSION OF THE FOLLOWING C++ CODE:

'''=============================================================================
   Copyright (c) 2018 Joel de Guzman. All rights reserved.

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================

IT CAN BE FOUND AT:

https://github.com/cycfi/bitstream_autocorrelation/blob/master/bcf2.cpp

'''

import numpy as np

# =============================================================================
# Utility functions
# =============================================================================

def smallest_pow2(n):
    m = 1
    while m < n:
        m <<= 1
    return m


# =============================================================================
# Zero-crossing detector
# =============================================================================

class ZeroCross:
    def __init__(self):
        self.y = 0

    def __call__(self, s):
        if s < -0.1:
            self.y = 0
        elif s > 0.0:
            self.y = 1
        return self.y


# =============================================================================
# Binary autocorrelation
# =============================================================================

def binary_autocorrelation(bits, start, end):
    """
    bits : numpy array of 0/1
    start, end : lag range (samples)

    Returns:
        corr : numpy array of mismatch counts
    """
    n = len(bits)
    corr = np.zeros(end, dtype=np.uint32)

    for shift in range(start, end):
        xor = bits[:n - shift] ^ bits[shift:]
        corr[shift] = np.count_nonzero(xor)

    return corr


# =============================================================================
# Main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    sps = 44100                       # samples per second
    min_freq = 50.0
    max_freq = 500.0
    freq = 261.626                   # test frequency (C4)

    period = sps / freq
    min_period = int(sps / max_freq)
    max_period = int(sps / min_freq)

    buff_size = smallest_pow2(int(np.ceil(max_period))) * 2

    # -------------------------------------------------------------------------
    # Generate test signal
    # -------------------------------------------------------------------------
    t = np.arange(buff_size)

    signal = (
        0.3 * np.sin(2 * np.pi * t / period) +
        0.4 * np.sin(4 * np.pi * t / period) +
        0.3 * np.sin(6 * np.pi * t / period)
    )

    # -------------------------------------------------------------------------
    # Zero-crossing → binary stream
    # -------------------------------------------------------------------------
    zc = ZeroCross()
    bits = np.array([zc(s) for s in signal], dtype=np.uint8)

    # -------------------------------------------------------------------------
    # Binary autocorrelation
    # -------------------------------------------------------------------------
    corr = binary_autocorrelation(bits, min_period, buff_size // 2)

    est_index = np.argmin(corr[min_period:]) + min_period
    min_count = corr[est_index]
    max_count = np.max(corr)

    # -------------------------------------------------------------------------
    # Harmonic rejection (avoid octave errors)
    # -------------------------------------------------------------------------
    sub_threshold = 0.15 * max_count
    max_div = est_index // min_period

    for div in range(max_div, 0, -1):
        all_strong = True
        for k in range(1, div):
            sub_period = int(k * est_index / div)
            if corr[sub_period] > sub_threshold:
                all_strong = False
                break
        if all_strong:
            est_index = int(est_index / div)
            break

    # -------------------------------------------------------------------------
    # Sub-sample zero-crossing refinement
    # -------------------------------------------------------------------------
    start_edge = np.where(signal > 0)[0][0]

    next_edge = start_edge + est_index
    while signal[next_edge] <= 0:
        next_edge += 1

    prev = signal[next_edge - 1]
    dy = signal[next_edge] - prev
    dx = -prev / dy

    n_samples = (next_edge - start_edge) + dx
    est_freq = sps / n_samples

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    print(f"Actual Frequency:    {freq:.3f} Hz")
    print(f"Estimated Frequency: {est_freq:.3f} Hz")
    print(f"Error: {1200 * np.log2(est_freq / freq):.2f} cents")
    print(f"Periodicity: {1.0 - (min_count / max_count):.3f}")


if __name__ == "__main__":
    main()

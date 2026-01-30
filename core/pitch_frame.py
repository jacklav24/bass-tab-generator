# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



from typing import Generator, Iterable
from core.framing import Frame
import numpy as np

class PitchFrame:
    def __init__(self, frame_index: int, f0_hz : float | None, confidence : float, method : str, raw_score: float | None):
        self.frame_index = frame_index
        self.f0_hz = f0_hz
        self.confidence = confidence
        self.method = method
        self.raw_score = raw_score
        
        
        
def estimate_pitch_autocorr(frame : Frame, sample_rate: int, f_min : float = 30.0, f_max : float = 500.0) -> PitchFrame:
    ''' 
    This function implements a real pitch estimation algorithm.
    '''
    samples = frame.samples
    
    # Remove DC offset
    samples = samples - np.mean(samples)
    
    # RMS energy check for silence
    rms_energy = np.sqrt(np.mean(samples**2))
    ENERGY_THRESHOLD = 1e-4
    
    if rms_energy < ENERGY_THRESHOLD:
        return PitchFrame(
            frame_index=frame.frame_index,
            f0_hz=None,
            confidence=0.0,
            method="autocorr",
            raw_score=None
        )
    
    # Copute autocorrelation
    autocorr = np.correlate(samples, samples, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only non-negative lags
    
    # Normalize by zero-lag value
    zero_lag = autocorr[0]
    autocorr = autocorr / zero_lag

    # Return no pitch if less than zero, since the signal is silent
    if zero_lag <= 0:
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    lag_min = int(np.ceil(sample_rate / f_max))
    lag_max = int(np.floor(sample_rate / f_min))
    
    # guard against invalid lag range
    if lag_min >= lag_max or lag_max >= len(autocorr):
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    search_region = autocorr[lag_min:lag_max]
    
    peaks = []
    for i in range(1, len(search_region) - 1):
        if search_region[i] > search_region[i - 1] and search_region[i] > search_region[i + 1]:
            peaks.append((i + lag_min, search_region[i]))  # (lag, value)
            
    if not peaks:
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    # select the strongest peak 
    # Select the strongest peak
    best_lag, best_value = max(peaks, key=lambda x: x[1])

    # Minimum peak strength threshold
    PEAK_THRESHOLD = 0.3

    if best_value < PEAK_THRESHOLD:
        return PitchFrame(
            frame_index=frame.frame_index,
            f0_hz=None,
            confidence=0.0,
            method="autocorr",
            raw_score=best_value,
        )

    # Convert lag to frequency
    f0_hz = sample_rate / best_lag

    # Confidence is proportional to normalized peak height
    confidence = float(best_value)

    return PitchFrame(
        frame_index=frame.frame_index,
        f0_hz=f0_hz,
        confidence=confidence,
        method="autocorr",
        raw_score=best_value,
    )
    



def estimate_pitch_sequence_autocorr(frames: Iterable[Frame], sample_rate: int, f_min: float = 30.0, f_max: float = 500.0,) -> Generator[PitchFrame, None, None]:
    for frame in frames:
        yield estimate_pitch_autocorr(frame, sample_rate, f_min, f_max)
        
        
def estimate_pitch_sequence(
    frames: Iterable[Frame],
    sample_rate: int,
    method: str = "autocorr",
    **kwargs,
    ) -> Generator[PitchFrame, None, None]:
    """
    Generic pitch estimation dispatcher.
    Routes frames to the selected estimator.
        """
    if method == "autocorr":
        return estimate_pitch_sequence_autocorr(frames, sample_rate, **kwargs)
    else:
        raise ValueError(f"Unknown pitch estimation method: {method}")
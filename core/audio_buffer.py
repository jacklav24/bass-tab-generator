# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



import soundfile as sf
import numpy as np


class AudioBuffer:
    def __init__(self, data: np.ndarray, sample_rate: int):
        self.data = data
        self.sample_rate = sample_rate
''' 
An `AudioBuffer` is a mono, float32, linear PCM signal sampled at a fixed rate (`sample_rate`).  
Its shape is `(num_samples, )`.
`num_samples / sample_rate` **always** defines time exactly.
No resampling, trimming, or normalization is a part of the AudioBuffer
'''
def load_audio_buffer(file_path: str) -> AudioBuffer:
    try:
        with sf.SoundFile(file_path) as f:
            data = f.read(dtype='float32', always_2d=False)
            sample_rate = f.samplerate
            
            # -- Check contract -- #
            
            # numpy array
            if not isinstance(data, np.ndarray):
                raise TypeError("Data must be a numpy.ndarray")
            
            # array of float32
            if data.dtype != np.float32:
                raise TypeError(f"Data must be of type float32, got {data.dtype}")
            
            # convert to mono if stereo
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            #must be mono
            if data.ndim!= 1:
                raise ValueError(f"Data must be 1-dimensional (mono), got {data.ndim} dimensions")
            
            #must have at least one sample
            if len(data) == 0:
                raise ValueError("Data must contain at least one sample")
            
            # must not contain NaNs or Infs
            if not np.all(np.isfinite(data)):
                raise ValueError("Data contains non-finite values")
            
            # sample rate must be positive
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"Sample rate must be a positive integer, got {sample_rate}")
            
            num_samples = len(data)
            duration_seconds = num_samples / sample_rate
            
            # positive duration 
            if duration_seconds <= 0:
                raise ValueError("Audio duration must be greater than zero")
            
            # -- End check contract -- #
            
            # Close the file
            f.close()
            return AudioBuffer(data, sample_rate)
            
            
    except Exception as e:
        print(f"An error occurred: {e}")
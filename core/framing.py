# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



from typing import Generator
from core.audio_buffer import AudioBuffer 
import numpy as np
'''
Frame {  
frame_index : int
start_sample : int
samples : np.ndarray
time_seconds : float 
}'''

class Frame:
    def __init__(self, frame_index: int, start_sample: int, samples: np.ndarray, time_seconds: float):
        self.frame_index = frame_index
        self.start_sample = start_sample
        self.samples = samples
        self.time_seconds = time_seconds


def build_frames(audio_buffer: AudioBuffer, frame_size_samples: int, hop_size_samples: int) -> Generator[Frame, None, None]:
    ''' Things to consider:
    The number of frames will depend on num_samples, frame_size_samples, hop_size_samples.
    We can compute the last valid frame index as:
        last_frame_index = floor((num_samples - frame_size_samples) / hop_size_samples)
    The last valid start sample is then:
        last_start_sample = last_frame_index * hop_size_samples
    '''
    last_frame_index = (len(audio_buffer.data) - frame_size_samples) // hop_size_samples

    
    for frame_index in range(last_frame_index + 1):
        start_sample = frame_index * hop_size_samples
        end_sample = start_sample + frame_size_samples
        samples = audio_buffer.data[start_sample:end_sample]
        time_seconds = start_sample / audio_buffer.sample_rate + frame_size_samples / audio_buffer.sample_rate / 2.0  # center time of the frame
        yield Frame(frame_index, start_sample, samples, time_seconds)
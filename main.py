# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



import core.audio_buffer as ab
import core.framing as fr
import analysis.pitch_frame as pf

def inspect_frames_around_time(frames, pitch_frames, target_time, window=0.25):
    print(f"\nInspecting frames around {target_time:.2f}s (±{window:.2f}s):\n")

    for f, p in zip(frames, pitch_frames):
        if abs(f.time_seconds - target_time) <= window:
            print(
                f"Frame {f.frame_index:5d} | "
                f"time={f.time_seconds:7.3f}s | "
                f"f0={p.f0_hz if p.f0_hz is not None else 'None':>7} | "
                f"conf={p.confidence:.3f}"
            )

def plot_pitch_window(frames, pitch_frames, t_start, t_end):
    import numpy as np
    import matplotlib.pyplot as plt

    times = []
    f0s = []
    confidences = []

    for f, p in zip(frames, pitch_frames):
        if t_start <= f.time_seconds <= t_end:
            times.append(f.time_seconds)
            f0s.append(p.f0_hz if p.f0_hz is not None else np.nan)
            confidences.append(p.confidence)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(times, f0s, ".", markersize=4)
    plt.ylabel("f0 (Hz)")
    plt.title(f"Pitch estimates from {t_start:.2f}s to {t_end:.2f}s")
    plt.ylim(0, 200)

    plt.subplot(2, 1, 2)
    plt.plot(times, confidences)
    plt.ylabel("Confidence")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def dummy_frame_function(audio, frames, frame_size_samples, duration_seconds):
    
    for f in frames[:5]:
        print(
            f"Frame {f.frame_index}: "
            f"start_sample={f.start_sample}, "
            f"time={f.time_seconds:.4f}s, "
            f"samples_shape={f.samples.shape}"
        )
        
    last = frames[-1]
    print(
        f"Last frame {last.frame_index}: "
        f"start_sample={last.start_sample}, "
        f"time={last.time_seconds:.4f}s"
    )

    end_time_estimate = (
        last.start_sample + frame_size_samples
    ) / audio.sample_rate

    print(f"End of last frame ≈ {end_time_estimate:.4f}s")
    print(f"Audio duration      = {duration_seconds:.4f}s")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(audio.data, alpha=0.5)
    for f in frames[:10]:
        plt.axvline(f.start_sample, color='r', alpha=0.3)
    plt.title("Audio with first few frame start positions")
    plt.show()
    

def dummy_pitch_frame_function(pitch_frames, frames):
    
    # Check 1: alignment
    assert len(pitch_frames) == len(frames), "PitchFrames and Frames length mismatch"

    # Check 2: frame index consistency
    for f, p in zip(frames[:10], pitch_frames[:10]):
        print(
            f"Frame {f.frame_index}: "
            f"time={f.time_seconds:.4f}s, "
            f"f0={p.f0_hz}, "
            f"confidence={p.confidence:.3f}"
        )
    last_frame = frames[-1]
    last_pitch = pitch_frames[-1]

    print(
        f"Last Frame {last_frame.frame_index}: "
        f"time={last_frame.time_seconds:.4f}s"
    )

    print(
        f"Last PitchFrame: "
        f"f0={last_pitch.f0_hz}, "
        f"confidence={last_pitch.confidence:.3f}"
    )
    times = [f.time_seconds for f in frames]
    f0s = [p.f0_hz for p in pitch_frames]
    confidences = [p.confidence for p in pitch_frames]
    import matplotlib.pyplot as plt
    import numpy as np

    f0s_plot = [f if f is not None else np.nan for f in f0s]

    plt.figure(figsize=(12, 4))
    plt.plot(times, f0s_plot, '.', markersize=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated f0 (Hz)")
    plt.title("Autocorrelation Pitch Estimates")
    plt.ylim(0, 600)
    plt.show()



def main():
    audio = ab.load_audio_buffer("./bass_files/bass1.wav")
    num_samples = len(audio.data)
    duration_seconds = num_samples / audio.sample_rate
    print(f"Loaded audio buffer with {len(audio.data)} samples at {audio.sample_rate} Hz for {duration_seconds:.2f} seconds")
    
    frame_size_samples = int(0.1 * audio.sample_rate)  # 100 ms frames
    hop_size_samples = int(0.025 * audio.sample_rate)   # 25 ms hop size
    
    frames = list(fr.build_frames(audio, frame_size_samples, hop_size_samples))
    print(f"Built {len(frames)} frames of size {frame_size_samples} samples with hop size {hop_size_samples}")
    
    pitch_frames = list(
    pf.estimate_pitch_sequence(
        frames,
        audio.sample_rate,
        method="autocorr",
        f_min=30.0,
        f_max=500.0,
    )
)

    print(f"Built {len(pitch_frames)} PitchFrames")
    
    # dummy_frame_function(audio, frames, frame_size_samples, duration_seconds)
    dummy_pitch_frame_function(pitch_frames, frames)
    plot_pitch_window(frames, pitch_frames, t_start=2.0, t_end=4.0)
    inspect_frames_around_time(frames, pitch_frames, target_time=2.5, window=0.3)



if __name__ == "__main__":
    main()

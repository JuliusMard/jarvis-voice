"""
This script is made for the intent of learning and can be utilized by anyone to check out how audio processing in python can be leveraged
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavf
import sounddevice as sd
import soundfile as sf


class SampleRateError(Exception):
    """Raised when an audio file's sample rate is below the required rate"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def record_audio(filename, duration=3, samplerate=16000, device='pipewire'):
    """Record a snippet of audio from the default device and save it as a WAV file."""
    print(f"Starting to record for {duration} seconds")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, device=device)
    sd.wait()
    wavf.write(f"{filename}.wav", samplerate, audio)
    audio = np.mean(a=audio, axis=(1))
    print(f"The recording was saved as {filename}.wav")
    return audio

def load_audio(filepath, samplerate=16000):
    """Returns an ndarray and samplerate made from the audio"""
    data, sr = sf.read(filepath)
    if sr >= samplerate:
        if data.ndim > 1:
            data = np.mean(a=data, axis=1)
        res = librosa.resample(y=data, orig_sr=sr, target_sr=samplerate)
    else:
        raise SampleRateError("Sorry, you need to increase the sampling rate of the audio")
    return res, samplerate

def compute_spectrogram(signal, samplerate=16000, frame_len=400, hop=160):
    r"""Compute a magnitude spectrogram with a short-time Fourier transform.

    The signal is cut into overlapping frames of ``frame_len`` samples, spaced
    ``hop`` samples apart. Each frame is then multiplied by a Hamming window and
    transformed with a real FFT:

    .. math::

        X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \, w[n] \, e^{-j \frac{2\pi k n}{N}}

    where :math:`N` is the frame length, :math:`H` the hop and :math:`w[n]`
    the window. The magnitude :math:`|X[m, k]|` is returned.

    Args:
        signal: 1D audio signal.
        samplerate: Sample rate of ``signal`` in Hz.
        frame_len: Frame length :math:`N` in samples.
        hop: Hop size :math:`H` in samples.

    Returns:
        frames: Magnitudes, shape ``(num_frames, frame_len // 2 + 1)``.
        frequencies: Frequency of each bin in Hz, :math:`f_k = \frac{k f_s}{N}`.
        times: Center time of each frame in seconds,
            :math:`t_m = \frac{mH + N/2}{f_s}`.
    """
    hamming = np.hamming(frame_len)
    M = 1 + (len(signal) - frame_len)//hop
    frames = []
    times = []
    frequencies = np.fft.rfftfreq(n=frame_len, d=1/samplerate)
    for m in range(M):
        frame_start = m * hop
        frame_end = frame_start + frame_len
        hammed_signal = signal[frame_start:frame_end] * hamming
        dft = np.fft.rfft(a=hammed_signal)
        tm = (m*hop + frame_len/2)/samplerate
        times.append(tm)
        frames.append(np.abs(dft))
    frames = np.array(frames)
    times = np.array(times)
    return frames, frequencies, times

def plot_spectrogram(frames, frequencies, times, title=None):
    """Plot a spectrogram using matplotlib"""
    x_labels = times
    y_labels = frequencies
    values_2d = frames.T
    values_2d = 20 * np.log10(values_2d + 1e-10)
    plt.pcolormesh(x_labels, y_labels, values_2d)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    if title is not None:
        plt.title(title)
    plt.colorbar(label="Magnitude [dB]")
    plt.show()

def melspectrogram(signal, samplerate=16000, hop_length=160):
    """Computes the melspectrogram given a signal in dB"""
    S = librosa.feature.melspectrogram(y=signal, sr=samplerate, hop_length=hop_length, n_fft=400, power=2, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def plot_melspectrogram(S, samplerate=16000, hop_length=160):
    """Creates a plot of the melspectrogram"""
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=samplerate, fmax=8000, ax=ax, hop_length=hop_length)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == "__main__":
    s = record_audio('recording')
    frames, frequencies, times = compute_spectrogram(signal=s)
    plot_spectrogram(frames=frames, frequencies=frequencies, times=times)
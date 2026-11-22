import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavf
import sounddevice as sd
import soundfile as sf


class SampleRateError(Exception):
    """"Raise an exception to the user that the sample rate is too low"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def record_audio(filename, duration=3, samplerate=16000, device='pipewire'):
    """
    Record a snippet of audio from default device and save it as an WAV file.
    """
    print(f"Starting to record for {duration} seconds")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, device=device)
    sd.wait()
    wavf.write(f"{filename}.wav", samplerate, audio)
    audio = np.mean(a=audio, axis=(1))
    print(f"The recoding was saved as {filename}.wav")
    return audio

def load_audio(filepath, samplerate=16000):
    data, sr = sf.read(filepath)
    if sr >= samplerate:
        if data.ndim > 1:
            data = np.mean(a=data, axis=1)
        res = librosa.resample(y=data, orig_sr=sr, target_sr=samplerate)
    else:
        raise SampleRateError("Sorry, you need to increase the sampling rate of the audio")
    return res, samplerate

def compute_spectrum(signal, samplerate=16000):
    dft = np.fft.rfft(a=signal)
    mag = np.abs(dft)
    frequencies = np.fft.rfftfreq(n=len(signal), d=1/samplerate)
    return frequencies, mag

def compute_spectrogram(signal, samplerate=16000, frame_len=400, hop=160):
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
    plt.savefig("figures/spectrogram.png", dpi=150)
    plt.show()

def melspectrogram(signal, sr=16000, hop_length=160):
    """
    Computes the melspectrogram given a signal in dB
    """
    S = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=400, power=2, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def plot_melspectrogram(S, sr=16000, hop_length=160):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, hop_length=hop_length)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == "__main__":
    s = record_audio('recording')
    frames, frequencies, times = compute_spectrogram(signal=s)
    plot_spectrogram(frames=frames, frequencies=frequencies, times=times)
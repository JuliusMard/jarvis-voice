import numpy as np
import scipy

from main import compute_spectrogram, load_audio, plot_spectrogram

if __name__ == "__main__":
    file = input("what file to test? ")
    res, sr = load_audio(file)
    f, t, Zxx = scipy.signal.stft(x=res, fs =sr, nperseg=400, noverlap=240, window='hamming')
    Zxx = np.abs(Zxx)
    plot_spectrogram(frames=Zxx.T, frequencies=f, times=t, title="scipy.signal.stft")
    frames, freq, times = compute_spectrogram(signal=res, samplerate=sr)
    plot_spectrogram(frames=frames, frequencies=freq, times=times, title="My STFT")
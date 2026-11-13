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
        res = librosa.resample(y=data, orig_sr=sr, target_sr=samplerate)
    else:
        raise SampleRateError("Sorry, you need to increase the sampling rate of the audio")
    return res, samplerate

def compute_spectrum(signal, samplerate=16000):
    dft = np.fft.rfft(a=signal)
    mag = np.abs(dft)
    frequencies = np.fft.rfftfreq(n=len(signal), d=1/samplerate)
    return frequencies, mag


if __name__ == "__main__":
    audio = record_audio("somefilev6")
    f, m = compute_spectrum(audio)
    plt.plot(f, m)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.show()
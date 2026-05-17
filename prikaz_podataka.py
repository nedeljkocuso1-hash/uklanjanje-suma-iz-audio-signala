import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Putanje do prvog uzorka
base = os.path.dirname(os.path.abspath(__file__))
clean_p = os.path.join(base, "dataset", "clean", "sample_0000.wav")
noisy_p = os.path.join(base, "dataset", "noisy", "sample_0000.wav")

def nacrtaj_prikaz(cp, np_path):
    y_c, sr = librosa.load(cp, sr=16000)
    y_n, _ = librosa.load(np_path, sr=16000)

    plt.figure(figsize=(12, 8))

    # Čist signal
    plt.subplot(2, 1, 1)
    D_c = librosa.amplitude_to_db(np.abs(librosa.stft(y_c)), ref=np.max)
    librosa.display.specshow(D_c, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spektrogram: Čist glas (Target)')
    plt.colorbar(format='%+2.0f dB')

    # Zašumljen signal
    plt.subplot(2, 1, 2)
    D_n = librosa.amplitude_to_db(np.abs(librosa.stft(y_n)), ref=np.max)
    librosa.display.specshow(D_n, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spektrogram: Glas + Šum (Input)')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig("prikaz_podataka.png")
    print("Slika 'prikaz_podataka.png' je sačuvana!")

nacrtaj_prikaz(clean_p, noisy_p)
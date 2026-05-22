import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

from generisanje import mix_audio

# Putanje do prvog uzorka
base = os.path.dirname(os.path.abspath(__file__))
clean_p = os.path.join(base, "dataset", "clean", "sample_0000.wav")
raw_data_dir = os.path.join(base, "raw_data")

def nacrtaj_prikaz1(cp, raw_dir):
    y_c, sr = librosa.load(cp, sr=16000)
    segment_samples=len(y_c)

    noise1_full, _=librosa.load(os.path.join(raw_dir, "sum1.wav"), sr=16000)
    noise2_full, _=librosa.load(os.path.join(raw_dir, "sum2.wav"), sr=16000)

    noise1_seg=noise1_full[:segment_samples]
    noise2_seg=noise2_full[:segment_samples]

    y_n1=mix_audio(y_c, noise1_seg, snr_db=5)
    y_n2=mix_audio(y_c, noise2_seg, snr_db=5)

    plt.figure(figsize=(12, 12))

    # Slika 1 :Čist signal
    plt.subplot(3, 1, 1)
    D_c = librosa.amplitude_to_db(np.abs(librosa.stft(y_c)), ref=np.max)
    librosa.display.specshow(D_c, sr=sr, x_axis='time', y_axis='hz')
    plt.title('1. Spektrogram: Čist glas (Target)')
    plt.colorbar(format='%+2.0f dB')

    # Slika 2 : Zašumljen signal
    plt.subplot(3, 1, 2)
    D_n1 = librosa.amplitude_to_db(np.abs(librosa.stft(y_n1)), ref=np.max)
    librosa.display.specshow(D_n1, sr=sr, x_axis='time', y_axis='hz')
    plt.title('2. Spektrogram: Glas + sum (sum1.wav)')
    plt.colorbar(format='%+2.0f dB')

    # Slika 3 : Zašumljen signal
    plt.subplot(3, 1, 3)
    D_n2 = librosa.amplitude_to_db(np.abs(librosa.stft(y_n2)), ref=np.max)
    librosa.display.specshow(D_n2, sr=sr, x_axis='time', y_axis='hz')
    plt.title('3. Spektrogram: Glas + sum (sum2.wav)')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig("prikaz_podataka.png", dpi=300)
    print("Slika 'prikaz_podataka.png' sa 3 spektograma je sačuvana!")
    plt.show()

def nacrtaj_prikaz2(cp, raw_dir):
    y_c, sr = librosa.load(cp, sr=16000)
    segment_samples=len(y_c)
    vreme=np.linspace(0,len(y_c)/sr, num=len(y_c))

    noise1_full, _=librosa.load(os.path.join(raw_dir, "sum1.wav"), sr=16000)
    noise2_full, _=librosa.load(os.path.join(raw_dir, "sum2.wav"), sr=16000)

    noise1_seg=noise1_full[:segment_samples]
    noise2_seg=noise2_full[:segment_samples]

    y_n1=mix_audio(y_c, noise1_seg, snr_db=5)
    y_n2=mix_audio(y_c, noise2_seg, snr_db=5)

    plt.figure(figsize=(12, 10))

    #Talas 1 :Čist signal
    plt.subplot(3, 1, 1)
    plt.plot(vreme, y_c, color='green', alpha=0.8)
    plt.title('1.Talasni oblik : cist glas', fontsize=12, fontweight='bold')
    plt.xlabel('Vreme (s)')
    plt.ylabel('Amplituda')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.5, 0.5)

    #Talas 2 : Glas + sum1
    plt.subplot(3, 1, 2)
    plt.plot(vreme, y_n1, color='blue', alpha=0.8)
    plt.title('2.Talasni oblik : Glas + sum1', fontsize=12, fontweight='bold')
    plt.xlabel('Vreme (s)')
    plt.ylabel('Amplituda')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.5, 0.5)

    #Talas 3 : Glas + sum2
    plt.subplot(3, 1, 3)
    plt.plot(vreme, y_n2, color='red', alpha=0.8)
    plt.title('3.Talasni oblik : Glas + sum2', fontsize=12, fontweight='bold')
    plt.xlabel('Vreme (s)')
    plt.ylabel('Amplituda')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.5, 0.5)
 
    plt.tight_layout()
    plt.savefig("prikaz_podataka_talas.png", dpi=300)
    print("Druga slika sa tri talasna oblika je uspesno kreirana")
    plt.show()

nacrtaj_prikaz1(clean_p, raw_data_dir)
nacrtaj_prikaz2(clean_p, raw_data_dir)
import librosa
import numpy as np
import soundfile as sf
import os
import random

# 1. APSOLUTNA PUTANJA (Ključ za Windows greške)
# Ovo pronalazi gde je PROJEKAT IS na disku
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Podešavanja putanja
DATA_DIR = os.path.join(BASE_PATH, "raw_data")
# Pravimo putanje za dataset
DATASET_DIR = os.path.join(BASE_PATH, "dataset")
OUTPUT_CLEAN = os.path.join(DATASET_DIR, "clean")
OUTPUT_NOISY = os.path.join(DATASET_DIR, "noisy")

SEGMENT_LEN = 3  # sekunde
SR = 16000      # kvalitet zvuka

def mix_audio(clean_seg, noise_seg, snr_db):
    p_clean = np.mean(clean_seg**2)
    p_noise = np.mean(noise_seg**2)
    if p_noise == 0: return clean_seg
    k = np.sqrt(p_clean / (p_noise * (10**(snr_db / 10))))
    return clean_seg + k * noise_seg

# 2. KREIRANJE FOLDERA (Jedan po jedan da izbegnemo WinError 3)
for d in [DATASET_DIR, OUTPUT_CLEAN, OUTPUT_NOISY]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        print(f"Napravljen folder: {d}")

# 3. UČITAVANJE
print("Učitavam fajlove iz raw_data...")
try:
    # Provera imena fajlova - moraju biti identični kao na slici
    clean_f_names = ['clean1.wav', 'clean2.wav']
    noise_f_names = ['sum1.wav', 'sum2.wav']

    clean_files = [librosa.load(os.path.join(DATA_DIR, f), sr=SR)[0] for f in clean_f_names]
    noise_files = [librosa.load(os.path.join(DATA_DIR, f), sr=SR)[0] for f in noise_f_names]

    all_clean = np.concatenate(clean_files)
    all_noise = np.concatenate(noise_files)

    segment_samples = SEGMENT_LEN * SR
    num_segments = len(all_clean) // segment_samples
    print(f"Generišem {num_segments} primera...")

    for i in range(num_segments):
        start = i * segment_samples
        clean_seg = all_clean[start : start + segment_samples]
        
        n_start = random.randint(0, len(all_noise) - segment_samples)
        noise_seg = all_noise[n_start : n_start + segment_samples]
        
        snr = random.uniform(0, 15)
        noisy_seg = mix_audio(clean_seg, noise_seg, snr)
        
        # Čuvanje
        filename = f"sample_{i:04d}.wav"
        sf.write(os.path.join(OUTPUT_CLEAN, filename), clean_seg, SR)
        sf.write(os.path.join(OUTPUT_NOISY, filename), noisy_seg, SR)
        
        if (i + 1) % 10 == 0: # Smanjio sam na 10 da brže vidiš progres
            print(f"Procesirano {i + 1}/{num_segments}...")

    print("\nUSPEH! Dataset je spreman.")
    print(f"Fajlovi su ovde: {DATASET_DIR}")

except Exception as e:
    print(f"\nGREŠKA: {e}")
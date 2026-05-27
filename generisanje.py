import shutil
import librosa
import numpy as np
import soundfile as sf
import os
import random

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_PATH, "raw_data")
DATASET_DIR = os.path.join(BASE_PATH, "dataset")
OUTPUT_CLEAN = os.path.join(DATASET_DIR, "clean")
OUTPUT_NOISY = os.path.join(DATASET_DIR, "noisy")
OUTPUT_TEST_CLEAN = os.path.join(DATASET_DIR, "test_clean")
OUTPUT_TEST_NOISY = os.path.join(DATASET_DIR, "test_noisy")

SEGMENT_LEN = 3
SR = 16000

def mix_audio(clean_seg, noise_seg, snr_db):
    p_clean = np.mean(clean_seg**2)
    p_noise = np.mean(noise_seg**2)
    if p_noise == 0:
        return clean_seg
    k = np.sqrt(p_clean / (p_noise * (10**(snr_db / 10))))
    return clean_seg + k * noise_seg

# Briše stari dataset
#for d in [OUTPUT_CLEAN, OUTPUT_NOISY, OUTPUT_TEST_CLEAN, OUTPUT_TEST_NOISY]:
#    if os.path.exists(d):
#        shutil.rmtree(d)
#        print(f"Obrisan stari folder: {d}")

for d in [DATASET_DIR, OUTPUT_CLEAN, OUTPUT_NOISY, OUTPUT_TEST_CLEAN, OUTPUT_TEST_NOISY]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        print(f"Napravljen folder: {d}")

print("Učitavam fajlove iz raw_data...")
try:
    clean_f_names = ['clean1.wav', 'clean2.wav', 'clean3.wav'] 
    noise_f_names = ['sum1.wav', 'sum2.wav', 'sum3.wav']

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

        filename = f"sample_{i:04d}.wav"
        sf.write(os.path.join(OUTPUT_CLEAN, filename), clean_seg, SR)
        sf.write(os.path.join(OUTPUT_NOISY, filename), noisy_seg, SR)

        if (i + 1) % 10 == 0:
            print(f"Procesirano {i + 1}/{num_segments}...")

    # Spaja test.wav i test2.wav u jedan test skup
    print("\nUčitavam test fajlove...")
    test1, _ = librosa.load(os.path.join(DATA_DIR, "test.wav"), sr=SR)
    test2, _ = librosa.load(os.path.join(DATA_DIR, "test2.wav"), sr=SR)  # ← novi test
    test_audio = np.concatenate([test1, test2])

    num_test_segments = len(test_audio) // segment_samples
    print(f"Generišem {num_test_segments} test primera...")

    for i in range(num_test_segments):
        start = i * segment_samples
        test_clean_seg = test_audio[start : start + segment_samples]

        n_start = random.randint(0, len(all_noise) - segment_samples)
        noise_seg = all_noise[n_start : n_start + segment_samples]

        snr = random.uniform(0, 15)
        test_noisy_seg = mix_audio(test_clean_seg, noise_seg, snr)

        test_filename = f"test_sample_{i:04d}.wav"
        sf.write(os.path.join(OUTPUT_TEST_CLEAN, test_filename), test_clean_seg, SR)
        sf.write(os.path.join(OUTPUT_TEST_NOISY, test_filename), test_noisy_seg, SR)

        if (i + 1) % 10 == 0:
            print(f"Test podaci: Procesirano {i+1}/{num_test_segments}...")

    print("\nUSPEH! Dataset i test set su spremni.")
    print(f"Fajlovi su ovde: {DATASET_DIR}")

except Exception as e:
    print(f"\nGREŠKA: {e}")
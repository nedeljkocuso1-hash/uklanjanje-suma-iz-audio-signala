import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sr=16000, n_fft=512, hop_length=256):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.file_names = os.listdir(clean_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Učitavanje
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)
        
        clean_wave, _ = librosa.load(clean_path, sr=self.sr)
        noisy_wave, _ = librosa.load(noisy_path, sr=self.sr)
        
        # Pretvaranje u spektrogram (STFT)
        clean_stft = librosa.stft(clean_wave, n_fft=self.n_fft, hop_length=self.hop_length)
        noisy_stft = librosa.stft(noisy_wave, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Uzimamo samo magnitudu (jačinu), ignorišemo fazu za početak
        clean_mag = np.abs(clean_stft)
        noisy_mag = np.abs(noisy_stft)
        
        # Dodajemo "kanal" dimenziju (kao kod slika: 1, visina, širina)
        clean_mag = clean_mag[np.newaxis, ...]
        noisy_mag = noisy_mag[np.newaxis, ...]
        
        return torch.FloatTensor(noisy_mag), torch.FloatTensor(clean_mag)

# Testiranje da li radi
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    dataset = AudioDataset(
        clean_dir=os.path.join(base, "dataset", "clean"),
        noisy_dir=os.path.join(base, "dataset", "noisy")
    )
    
    # Uzmi jedan primer
    input_data, target_data = dataset[0]
    print(f"Oblik ulaza (noisy): {input_data.shape}")
    print(f"Oblik cilja (clean): {target_data.shape}")
    print("DataLoader uspešno učitava podatke!")
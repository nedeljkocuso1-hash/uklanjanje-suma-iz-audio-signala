import torch
import torch.nn as nn
import torchaudio
import os

# Putanja do dataset-a (relativna, prema strukturi projekta)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_DIR = os.path.join(BASE_PATH, "ocisceni_test_fajlovi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_PATH, "final_unet_model.pth")

# =====================================================================
# DEFINICIJA ARHITEKTURE MODELA
# =====================================================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)

        u2 = self.up2(b)
        if u2.shape != e2.shape:
            u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2:])
        merge2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(merge2)

        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2:])
        merge1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(merge1)
        return self.final_conv(d1)


class TestAudioDataset:
    def __init__(self, data_dir):
        self.noisy_dir = os.path.join(data_dir, "test_noisy")
        self.clean_dir = os.path.join(data_dir, "test_clean")
        self.filenames = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith('.wav')])

    def __len__(self):
        return len(self.filenames)

    def get_wav_and_spectrogram(self, idx):
        filename = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy_waveform, sr = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)

        n_fft = 512
        hop_length = 256

        noisy_spec = torch.stft(noisy_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        clean_spec = torch.stft(clean_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)

        return noisy_waveform, clean_waveform, noisy_spec, clean_spec, filename, sr


def pokreni_testiranje():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Greška: Fajl sa težinama modela nije pronađen na putanji: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_set = TestAudioDataset(DATASET_PATH)
    if len(test_set) == 0:
        print("Greška: Nema .wav fajlova u test folderima.")
        return

    print(f"Pokrećemo evaluaciju na {len(test_set)} testnih fajlova...")

    criterion = torch.nn.MSELoss()
    ukupni_test_loss = 0.0
    fajlovi_za_cuvanje = 3

    with torch.no_grad():
        for i in range(len(test_set)):
            noisy_wav, clean_wav, noisy_spec, clean_spec, filename, sr = test_set.get_wav_and_spectrogram(i)
            noisy_mag = torch.abs(noisy_spec).unsqueeze(0).to(device)
            clean_mag = torch.abs(clean_spec).to(device)

            izlaz_mag = model(noisy_mag).squeeze(0)
            loss = criterion(izlaz_mag, clean_mag)
            ukupni_test_loss += loss.item()

            if i < fajlovi_za_cuvanje:
                faza = torch.angle(noisy_spec).to(device)
                rekonstruisani_spektrogram = izlaz_mag * torch.exp(1j * faza)

                n_fft = 512
                hop_length = 256
                ocisceni_wav = torch.istft(rekonstruisani_spektrogram, n_fft=n_fft, hop_length=hop_length, length=noisy_wav.shape[-1])

                ocisceni_wav_cpu = ocisceni_wav.cpu()
                izlazna_putanja = os.path.join(OUTPUT_DIR, f"ociscen_{filename}")
                torchaudio.save(izlazna_putanja, ocisceni_wav_cpu, sr)
                print(f"✓ Uspešno očišćen i sačuvan audio: ociscen_{filename}")

    prosecni_test_loss = ukupni_test_loss / len(test_set)
    print(f"\n=========================================")
    print(f"KONAČNI REZULTAT NA TEST SKUPU:")
    print(f"Prosečan Test Loss (MSE): {prosecni_test_loss:.6f}")
    print(f"=========================================")


if __name__ == "__main__":
    pokreni_testiranje()
# uklanjanje-suma-iz-audio-signala

U-Net model za uklanjanje šuma iz govornih snimaka. Model prima zašumljeni audio signal i rekonstruiše očišćen signal.

---

## Instalacija

Projekat koristi `uv` za upravljanje zavisnostima.

**1. Instaliraj uv:**
```bash
pip install uv
```

**2. Kloniraj repozitorijum:**
```bash
git clone <url_repozitorijuma>
cd "Projekat IS"
```

**3. Instaliraj zavisnosti:**
```bash
uv sync
```

---

## Vodič za pokretanje

### Korak 1 — Priprema sirovih fajlova

U folder `raw_data/` postavi sledeće fajlove:
- `clean1.wav`, `clean2.wav`, `clean3.wav` — čisti govorni snimci
- `sum1.wav`, `sum2.wav` — snimci šuma
- `test.wav`, `test2.wav` — govorni snimci za testiranje (ne koriste se u treningu)

---

### Korak 2 — Generisanje dataseta

```bash
uv run generisanje.py
```

Šta radi:
- Seče govorne snimke na segmente od 3 sekunde
- Meša čist glas sa šumom na različitim nivoima (SNR 0-15 dB)
- Kreira foldere `dataset/clean`, `dataset/noisy`, `dataset/test_clean`, `dataset/test_noisy`

> **Napomena:** Ako pokrećeš ponovo, ručno obrišu sadržaj `dataset/` foldera pre pokretanja da ne dođe do mešanja starih i novih fajlova.

---

### Korak 3 — Prikaz podataka (opciono)

```bash
uv run prikaz_podataka.py
```

Generiše dve slike:
- `prikaz_podataka.png` — spektrogrami čistog i zašumljenog signala
- `prikaz_podataka_talas.png` — talasni oblici čistog i zašumljenog signala

---

### Korak 4 — Trening (Google Colab)

Trening se pokreće na Google Colab-u zbog GPU podrške.

1. Idi na [colab.research.google.com](https://colab.research.google.com)
2. Uploaduj ili otvori `main.py`
3. Uploaduj `dataset/` folder na Google Drive u folder `U-Net_Projekat/`
4. Pokreni sve ćelije redom
5. Model se automatski čuva kao `final_unet_model.pth` na Google Drive-u

Tokom treninga ispisuje se loss po epohama i na kraju se generiše grafik `train_val_loss_70_15_15.png`.

---

### Korak 5 — Evaluacija (Google Colab)

1. Otvori `test.py` na Colabu
2. Pokreni sve ćelije
3. Ispisuje se finalni Test Loss (MSE)
4. Prva 3 očišćena audio fajla se čuvaju u `ocisceni_test_fajlovi/` na Google Drive-u

---

## Struktura projekta
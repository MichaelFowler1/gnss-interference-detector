"""
Generates docs/hero.png — a single 'gist' image for the README.
Shows a real model detection: STFT spectrogram + PSD + verdict banner.
Run: .\.venv\Scripts\python.exe make_hero.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sp

from train_pipeline import CognitiveEWNet
from realistic_rf_env import RealisticRFEnvironment

CLASS = {0: ("CLEAR", "NONE"), 1: ("JAMMED", "BARRAGE"), 2: ("JAMMED", "TONE")}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CognitiveEWNet(num_classes=3)
model.load_state_dict(torch.load("cognitive_ew_model.pth", map_location=device))
model.to(device).eval()

# Build a compelling, unambiguous TONE-jamming scene
env = RealisticRFEnvironment(sample_rate=1e6, num_symbols=1024)
base = env.generate_base_qpsk()
corrupted = env.apply_channel_effects(base, cfo_hz=8000, snr_db=15)
iq = env.apply_tone_jamming(corrupted, jsr_db=18, offset_hz=-120e3)

# --- run the real model ---
iq_t = torch.from_numpy(iq).to(torch.complex64).to(device)
stft = torch.stft(iq_t, n_fft=128, hop_length=32, return_complex=True)
spec_db = (20 * torch.log10(torch.abs(stft) + 1e-6))
with torch.no_grad():
    logits = model(spec_db.unsqueeze(0).unsqueeze(0))
    probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()
    cls = int(probs.argmax())
conf = float(probs[cls])
status, atype = CLASS[cls]

# --- figure ---
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#d7e2f0",
                     "axes.edgecolor": "#1b2740"})
fig = plt.figure(figsize=(12, 6.2), facecolor="#070b12")
gs = GridSpec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
              hspace=0.35, wspace=0.22, left=0.06, right=0.97, top=0.80, bottom=0.10)

# Title / banner
fig.text(0.06, 0.93, "COGNITIVE EW  ·  RF INTERFERENCE CLASSIFIER",
         fontsize=17, fontweight="bold", color="#d7e2f0")
fig.text(0.06, 0.875, "Edge-AI node: raw I/Q  →  GPU STFT  →  CNN  →  jamming verdict",
         fontsize=11, color="#6b7d95")
banner = "#ff4d4d" if status == "JAMMED" else "#25d07d"
fig.text(0.97, 0.915, f"{status} · {atype}", ha="right", fontsize=20,
         fontweight="bold", color=banner)
fig.text(0.97, 0.865, f"confidence {conf*100:.1f}%   |   {str(device).upper()}",
         ha="right", fontsize=11, color="#6b7d95")

# Spectrogram (spans left column, both rows)
axs = fig.add_subplot(gs[:, 0], facecolor="#060a11")
S = spec_db.cpu().numpy()
axs.imshow(np.fft.fftshift(S, axes=0), aspect="auto", origin="lower",
           cmap="viridis", extent=[0, S.shape[1], -500, 500])
axs.set_title("Live Spectrogram (STFT magnitude, dB)", fontsize=11, color="#9fb2c9", loc="left")
axs.set_xlabel("Time frames", fontsize=9); axs.set_ylabel("Frequency (kHz)", fontsize=9)
axs.tick_params(colors="#6b7d95", labelsize=8)

# PSD (top right)
axp = fig.add_subplot(gs[0, 1], facecolor="#060a11")
f, pxx = sp.welch(iq, env.fs, return_onesided=False, nperseg=256)
f = np.fft.fftshift(f) / 1e3; pxx = 10*np.log10(np.fft.fftshift(pxx)+1e-12)
axp.plot(f, pxx, color="#39c2ff", lw=1.4)
axp.set_title("Power Spectral Density", fontsize=11, color="#9fb2c9", loc="left")
axp.set_xlabel("kHz", fontsize=9); axp.set_ylabel("dB/Hz", fontsize=9)
axp.tick_params(colors="#6b7d95", labelsize=8); axp.grid(alpha=0.15)

# Class probabilities (bottom right)
axb = fig.add_subplot(gs[1, 1], facecolor="#060a11")
labels = ["CLEAR", "BARRAGE", "TONE"]; cols = ["#25d07d", "#ff4d4d", "#ffb020"]
axb.barh(labels, probs*100, color=cols)
axb.set_xlim(0, 100); axb.invert_yaxis()
axb.set_title("Class probabilities (%)", fontsize=11, color="#9fb2c9", loc="left")
axb.tick_params(colors="#6b7d95", labelsize=9)
for i, p in enumerate(probs):
    axb.text(min(p*100+2, 92), i, f"{p*100:.1f}", va="center", fontsize=9, color="#d7e2f0")

os.makedirs("docs", exist_ok=True)
out = os.path.join("docs", "hero.png")
fig.savefig(out, dpi=140, facecolor="#070b12")
print(f"[+] wrote {out}  ->  verdict {status}/{atype} @ {conf*100:.1f}%")

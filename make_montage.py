"""
Generates docs/montage.png — clear / barrage / tone side-by-side spectrograms,
each with the model's real verdict. Secondary README image.
Run: .\.venv\Scripts\python.exe make_montage.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_pipeline import CognitiveEWNet
from realistic_rf_env import RealisticRFEnvironment

CLASS = {0: ("CLEAR", "NONE"), 1: ("JAMMED", "BARRAGE"), 2: ("JAMMED", "TONE")}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CognitiveEWNet(num_classes=3)
model.load_state_dict(torch.load("cognitive_ew_model.pth", map_location=device))
model.to(device).eval()


def scene(kind):
    env = RealisticRFEnvironment(sample_rate=1e6, num_symbols=1024)
    x = env.apply_channel_effects(env.generate_base_qpsk(), cfo_hz=8000, snr_db=15)
    if kind == 1:
        x = env.apply_barrage_jamming(x, jsr_db=18)
    elif kind == 2:
        x = env.apply_tone_jamming(x, jsr_db=18, offset_hz=-120e3)
    return x


def predict(iq):
    t = torch.from_numpy(iq).to(torch.complex64).to(device)
    s = 20 * torch.log10(torch.abs(torch.stft(t, 128, 32, return_complex=True)) + 1e-6)
    with torch.no_grad():
        p = torch.softmax(model(s.unsqueeze(0).unsqueeze(0)), 1).squeeze(0).cpu().numpy()
    return s.cpu().numpy(), int(p.argmax()), float(p.max())


plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#d7e2f0"})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="#070b12")
fig.subplots_adjust(left=0.05, right=0.98, top=0.68, bottom=0.12, wspace=0.18)
fig.text(0.05, 0.93, "COGNITIVE EW  ·  THREE SPECTRUM STATES, ONE MODEL",
         fontsize=15, fontweight="bold")
fig.text(0.05, 0.875, "Same CNN classifies a clean link and two distinct electronic attacks in real time",
         fontsize=10, color="#6b7d95")

for ax, kind in zip(axes, [0, 1, 2]):
    iq = scene(kind)
    S, cls, conf = predict(iq)
    ax.imshow(np.fft.fftshift(S, axes=0), aspect="auto", origin="lower",
              cmap="viridis", extent=[0, S.shape[1], -500, 500])
    status, atype = CLASS[cls]
    col = "#25d07d" if status == "CLEAR" else "#ff4d4d"
    ax.set_title(f"{status} · {atype}\n{conf*100:.1f}% confidence",
                 fontsize=12, fontweight="bold", color=col)
    ax.set_xlabel("Time frames", fontsize=8)
    ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.tick_params(colors="#6b7d95", labelsize=7)

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/montage.png", dpi=140, facecolor="#070b12")
print("[+] wrote docs/montage.png")

# Cognitive EW — RF Interference Classifier

A GPU-accelerated edge-AI node that ingests raw radio I/Q samples, converts them
to a spectrogram on the GPU, and classifies the RF spectrum in real time as
**clear**, **barrage-jammed**, or **tone-jammed** — a core building block for
electronic-warfare (EW) situational awareness.

![Cognitive EW RF Interference Classifier](docs/hero.png)

## What it does

Raw I/Q  →  GPU STFT  →  CNN  →  jamming verdict, at low latency on the edge.

| Class | Meaning |
|-------|---------|
| `CLEAR / NONE` | Nominal link (with realistic Doppler, multipath fading, thermal noise) |
| `JAMMED / BARRAGE` | Wideband noise attack — raises the whole noise floor |
| `JAMMED / TONE` | Narrowband continuous-wave attack — a sharp spectral spike |

The same model handles all three states:

![Three spectrum states, one model](docs/montage.png)

## How it works

1. **`realistic_rf_env.py`** — synthesizes high-fidelity QPSK I/Q with channel
   impairments (CFO/Doppler, Rayleigh multipath, AWGN) and adversarial jamming
   profiles (barrage, tone, receiver clipping).
2. **`train_pipeline.py`** — defines `CognitiveEWNet` (a CNN over log-magnitude
   STFT spectrograms) and the training loop.
3. **`train_and_save.py`** — trains and writes `cognitive_ew_model.pth`.
4. **`inference_node.py`** — FastAPI microservice exposing `POST /predict`;
   runs the STFT + inference on the GPU.
5. **`live_server.py`** — a live browser dashboard (spectrogram, PSD, rolling
   accuracy) for demos.

## Quick start

```powershell
# From the project root, using the bundled virtual environment
.\.venv\Scripts\python.exe inference_node.py      # REST API  -> http://127.0.0.1:8000/docs
.\.venv\Scripts\python.exe test_client.py         # streams a jammed signal and prints the verdict
.\.venv\Scripts\python.exe live_server.py         # live dashboard -> http://127.0.0.1:8010/
```

Example response from `POST /predict`:

```json
{ "status": "JAMMED", "type": "TONE", "confidence": 0.9801, "hardware_accelerated": true }
```

## Regenerating the README images

```powershell
.\.venv\Scripts\python.exe make_hero.py       # -> docs/hero.png
.\.venv\Scripts\python.exe make_montage.py    # -> docs/montage.png
```

## Requirements

- Python 3.12, PyTorch (CUDA build), FastAPI, uvicorn, NumPy, SciPy, matplotlib
- An NVIDIA GPU is used automatically when available; otherwise it falls back to CPU.

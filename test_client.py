# test_client.py
import requests
import numpy as np
from realistic_rf_env import RealisticRFEnvironment

print("[*] Generating an adversarial live RF stream...")
# 1. Simulate an environment experiencing Tone Jamming
env = RealisticRFEnvironment(sample_rate=1e6, num_symbols=1024)
base_qpsk = env.generate_base_qpsk()
corrupted = env.apply_channel_effects(base_qpsk, cfo_hz=8000, snr_db=15)
jammed = env.apply_tone_jamming(corrupted, jsr_db=18, offset_hz=-120e3) # Jammer at -120kHz

# 2. Flatten complex numbers into a standard list [I0, Q0, I1, Q1...] for JSON transport
flat_iq = []
for sample in jammed:
    flat_iq.extend([float(sample.real), float(sample.imag)])

print("[*] Streaming raw I/Q array to the RTX 3080 Inference Node...")
# 3. POST the raw data to your FastAPI server
try:
    response = requests.post("http://127.0.0.1:8000/predict", json={"iq_data": flat_iq})
    print("\n[+] LIVE TELEMETRY ALERT RECEIVED FROM EDGE NODE:")
    print(response.json())
except Exception as e:
    print(f"[-] Connection Failed: {e}")
"""
Live browser dashboard for the Cognitive EW Inference Node.

Serves a single page at http://127.0.0.1:8010/ that polls /simulate once per
second. Each poll generates a fresh, random RF scenario (clear / barrage / tone),
runs it through the same GPU model used by inference_node.py, and returns the
classification plus a spectrogram + power-spectrum for live visualization.

Run:  .\.venv\Scripts\python.exe live_server.py
"""
import random

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scipy import signal as sp_signal

from train_pipeline import CognitiveEWNet
from realistic_rf_env import RealisticRFEnvironment

app = FastAPI(title="Cognitive EW Live Dashboard")

CLASS_MAP = {
    0: {"status": "CLEAR", "type": "NONE"},
    1: {"status": "JAMMED", "type": "BARRAGE"},
    2: {"status": "JAMMED", "type": "TONE"},
}

model = None
device = None


@app.on_event("startup")
def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Live dashboard initializing on: {device}")
    model = CognitiveEWNet(num_classes=3)
    model.load_state_dict(torch.load("cognitive_ew_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("[+] Model loaded. Open http://127.0.0.1:8010/")


def _make_scenario():
    """Randomly pick a ground-truth scenario and synthesize live I/Q."""
    env = RealisticRFEnvironment(sample_rate=1e6, num_symbols=1024)
    base = env.generate_base_qpsk()
    cfo = random.uniform(-15000, 15000)
    snr = random.uniform(10, 22)
    corrupted = env.apply_channel_effects(base, cfo_hz=cfo, snr_db=snr)

    truth = random.choice([0, 1, 2])
    if truth == 1:
        iq = env.apply_barrage_jamming(corrupted, jsr_db=random.uniform(12, 22))
    elif truth == 2:
        offset = random.choice([-1, 1]) * random.uniform(80e3, 200e3)
        iq = env.apply_tone_jamming(corrupted, jsr_db=random.uniform(12, 22), offset_hz=offset)
    else:
        iq = corrupted
    return iq, truth, env.fs


def _predict(iq_complex):
    iq_tensor = torch.from_numpy(iq_complex).to(torch.complex64).to(device)
    stft = torch.stft(iq_tensor, n_fft=128, hop_length=32, return_complex=True)
    spec = torch.abs(stft)
    spec_db = 20 * torch.log10(spec + 1e-6)
    inp = spec_db.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        conf, cls = torch.max(probs, dim=1)
    return int(cls.item()), float(conf.item()), probs.squeeze(0).cpu().numpy(), spec_db.cpu().numpy()


@app.get("/simulate")
def simulate():
    iq, truth, fs = _make_scenario()
    cls, conf, probs, spec_db = _predict(iq)

    # Downsample spectrogram to a light heatmap for the browser
    freq_bins, time_bins = spec_db.shape
    t_step = max(1, time_bins // 96)
    heat = spec_db[:, ::t_step]
    hmin, hmax = float(heat.min()), float(heat.max())

    # Power spectral density (welch), centered
    f, pxx = sp_signal.welch(iq, fs, return_onesided=False, nperseg=256)
    f = np.fft.fftshift(f) / 1e3
    pxx = 10 * np.log10(np.fft.fftshift(pxx) + 1e-12)

    return {
        "status": CLASS_MAP[cls]["status"],
        "type": CLASS_MAP[cls]["type"],
        "confidence": round(conf, 4),
        "predicted": cls,
        "truth": truth,
        "truth_label": CLASS_MAP[truth]["type"],
        "correct": cls == truth,
        "probs": [round(float(p), 4) for p in probs],
        "hardware": str(device),
        "psd": {"f": f.round(1).tolist(), "p": pxx.round(2).tolist()},
        "spec": {"min": hmin, "max": hmax, "rows": heat.round(1).tolist()},
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cognitive EW — Live Spectrum Monitor</title>
<style>
  :root { --bg:#070b12; --panel:#0e1522; --ink:#d7e2f0; --dim:#6b7d95;
          --clear:#25d07d; --jam:#ff4d4d; --line:#39c2ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:"Segoe UI",system-ui,sans-serif; }
  header { padding:16px 22px; border-bottom:1px solid #1b2740;
           display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:18px; margin:0; letter-spacing:.5px; }
  header .sub { color:var(--dim); font-size:12px; }
  .wrap { display:grid; grid-template-columns:340px 1fr; gap:18px; padding:18px 22px; }
  .card { background:var(--panel); border:1px solid #1b2740; border-radius:12px; padding:16px; }
  .status { font-size:44px; font-weight:800; letter-spacing:1px; }
  .type { font-size:16px; color:var(--dim); margin-top:2px; }
  .conf { font-size:13px; color:var(--dim); margin-top:12px; }
  .bar { height:10px; background:#16233b; border-radius:6px; overflow:hidden; margin-top:6px; }
  .bar > div { height:100%; width:0; transition:width .3s; }
  .row { display:flex; justify-content:space-between; font-size:12px;
         color:var(--dim); margin-top:10px; }
  .pill { padding:3px 8px; border-radius:20px; font-size:11px; font-weight:700; }
  .ok { background:rgba(37,208,125,.15); color:var(--clear); }
  .bad { background:rgba(255,77,77,.15); color:var(--jam); }
  canvas { width:100%; display:block; border-radius:8px; background:#060a11; }
  h3 { font-size:12px; text-transform:uppercase; letter-spacing:1px;
       color:var(--dim); margin:0 0 10px; }
  .probs span { display:inline-block; width:100%; }
</style>
</head>
<body>
<header>
  <h1>COGNITIVE EW — LIVE SPECTRUM MONITOR</h1>
  <span class="sub" id="hw">initializing…</span>
</header>
<div class="wrap">
  <div class="card">
    <div class="status" id="status">—</div>
    <div class="type" id="type">standby</div>
    <div class="conf">Confidence <span id="confv">0%</span></div>
    <div class="bar"><div id="confbar"></div></div>

    <h3 style="margin-top:22px">Class probabilities</h3>
    <div class="probs" id="probs"></div>

    <div class="row"><span>Ground truth</span><span id="truth">—</span></div>
    <div class="row"><span>Model verdict</span><span id="verdict" class="pill">—</span></div>
    <div class="row"><span>Rolling accuracy</span><span id="acc">—</span></div>
  </div>

  <div>
    <div class="card" style="margin-bottom:18px">
      <h3>Live Spectrogram (STFT magnitude, dB)</h3>
      <canvas id="spec" width="900" height="220"></canvas>
    </div>
    <div class="card">
      <h3>Power Spectral Density (dB/Hz vs kHz)</h3>
      <canvas id="psd" width="900" height="180"></canvas>
    </div>
  </div>
</div>

<script>
const LABELS = ["CLEAR/NONE","BARRAGE","TONE"];
const COL = ["#25d07d","#ff4d4d","#ffb020"];
let hits=0, total=0;

function heat(v){ // v in 0..1 -> viridis-ish
  const r=Math.round(255*Math.min(1,Math.max(0,1.4*v-0.3)));
  const g=Math.round(255*Math.min(1,Math.max(0,1.2*v)));
  const b=Math.round(255*Math.min(1,Math.max(0,0.9-0.9*v)+0.3*v));
  return `rgb(${r},${g},${b})`;
}

function drawSpec(spec){
  const c=document.getElementById('spec'), x=c.getContext('2d');
  const rows=spec.rows, R=rows.length, C=rows[0].length;
  const span=(spec.max-spec.min)||1;
  const cw=c.width/C, ch=c.height/R;
  for(let i=0;i<R;i++){
    const rr=rows[R-1-i];
    for(let j=0;j<C;j++){
      x.fillStyle=heat((rr[j]-spec.min)/span);
      x.fillRect(j*cw,i*ch,cw+1,ch+1);
    }
  }
}

function drawPsd(psd){
  const c=document.getElementById('psd'), x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const p=psd.p, n=p.length;
  let lo=Math.min(...p), hi=Math.max(...p); const sp=(hi-lo)||1;
  x.strokeStyle='#39c2ff'; x.lineWidth=1.5; x.beginPath();
  for(let i=0;i<n;i++){
    const xx=i/(n-1)*c.width;
    const yy=c.height-((p[i]-lo)/sp)*(c.height-10)-5;
    i?x.lineTo(xx,yy):x.moveTo(xx,yy);
  }
  x.stroke();
}

async function tick(){
  try{
    const r=await fetch('/simulate'); const d=await r.json();
    const jam=d.status==='JAMMED';
    document.getElementById('hw').textContent='hardware: '+d.hardware;
    const st=document.getElementById('status');
    st.textContent=d.status; st.style.color=jam?'var(--jam)':'var(--clear)';
    document.getElementById('type').textContent=jam?('Attack type: '+d.type):'spectrum nominal';
    document.getElementById('confv').textContent=Math.round(d.confidence*100)+'%';
    const cb=document.getElementById('confbar');
    cb.style.width=Math.round(d.confidence*100)+'%';
    cb.style.background=jam?'var(--jam)':'var(--clear)';

    document.getElementById('probs').innerHTML=d.probs.map((p,i)=>
      `<div class="row" style="margin-top:6px"><span style="color:${COL[i]}">${LABELS[i]}</span>`
      +`<span>${Math.round(p*100)}%</span></div>`
      +`<div class="bar"><div style="width:${Math.round(p*100)}%;background:${COL[i]}"></div></div>`
    ).join('');

    document.getElementById('truth').textContent=d.truth_label;
    const v=document.getElementById('verdict');
    v.textContent=d.correct?'CORRECT':'MISS';
    v.className='pill '+(d.correct?'ok':'bad');
    total++; if(d.correct) hits++;
    document.getElementById('acc').textContent=Math.round(100*hits/total)+'%  ('+hits+'/'+total+')';

    drawSpec(d.spec); drawPsd(d.psd);
  }catch(e){ document.getElementById('hw').textContent='connection lost — is live_server running?'; }
}
setInterval(tick, 1000); tick();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)

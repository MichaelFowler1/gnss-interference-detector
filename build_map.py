import json, os
gj=open("theater.geojson").read()
data=json.loads(gj)
st=data["stats"]

# optional classification labels (from classify.py live or --demo)
labels={}; prov=None
if os.path.exists("labels.json"):
    L=json.load(open("labels.json")); prov=L.get("provenance")
    for x in L["labels"]:
        labels[f"{int(round(x['lat']*100))}_{int(round(x['lon']*100))}"]=x["label"]
labels_js=json.dumps(labels,separators=(",",":"))

badge = ('<div class="badge live">REAL DATA · __DATE__</div>' if not prov else
 '<div class="badge illus">ILLUSTRATIVE CLASSIFICATION · region-based</div>' if prov=="illustrative" else
 '<div class="badge live">LIVE CLASSIFICATION · OpenSky</div>')

html=r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GNSS Interference — Jamming vs Spoofing</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#0a131c;--panel:#0f1d2a;--panel2:#11212f;--line:#1b3346;--txt:#cdd9e3;--muted:#6f8499;
--faint:#445b70;--clean:#1fb8a6;--elev:#f59e0b;--jam:#e63950;--spoof:#b14dff;--mixed:#ff8a3d;--unres:#cf7a4a;--cyan:#38bdf8;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);overflow:hidden}
#app{display:grid;grid-template-columns:1fr 360px;grid-template-rows:auto 1fr;height:100vh}
header{grid-column:1/3;display:flex;align-items:center;gap:14px;padding:10px 16px;
background:linear-gradient(180deg,#0c1825,#0a131c);border-bottom:1px solid var(--line)}
.brand{display:flex;flex-direction:column;line-height:1.15}
.brand b{font-family:var(--mono);font-size:14px;letter-spacing:.13em;color:#eaf2f8}
.brand span{font-size:10.5px;color:var(--muted)}
.badge{margin-left:auto;font-family:var(--mono);font-size:10px;letter-spacing:.06em;padding:5px 10px;
border-radius:4px;display:flex;align-items:center;gap:7px}
.badge::before{content:"";width:7px;height:7px;border-radius:50%}
.badge.live{color:#bdf5ea;background:rgba(31,184,166,.12);border:1px solid rgba(31,184,166,.4)}
.badge.live::before{background:var(--clean);box-shadow:0 0 8px var(--clean)}
.badge.illus{color:#ffd9a0;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.4)}
.badge.illus::before{background:var(--elev);box-shadow:0 0 8px var(--elev)}
#map{grid-column:1;grid-row:2;background:#0a131c}.leaflet-container{background:#0a131c;font-family:var(--mono)}
aside{grid-column:2;grid-row:2;background:var(--panel);border-left:1px solid var(--line);overflow-y:auto}
.sec{padding:13px 15px;border-bottom:1px solid var(--line)}
.sec h2{margin:0 0 9px;font-family:var(--mono);font-size:10px;letter-spacing:.13em;color:var(--muted);text-transform:uppercase}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:8px 10px}
.stat .n{font-family:var(--mono);font-size:18px;font-weight:600}.stat .l{font-size:10px;color:var(--muted)}
.stat.jam .n{color:var(--jam)}.stat.spoof .n{color:var(--spoof)}.stat.elev .n{color:var(--elev)}.stat.clean .n{color:var(--clean)}
.feed{display:flex;flex-direction:column;gap:6px}
.det{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--unres);
border-radius:4px;padding:8px 10px;cursor:pointer}.det:hover{background:#16293a}
.det.jam{border-left-color:var(--jam)}.det.spoof{border-left-color:var(--spoof)}.det.mixed{border-left-color:var(--mixed)}
.det .top{display:flex;justify-content:space-between;align-items:baseline}
.det .cls{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.08em}
.det.jam .cls{color:var(--jam)}.det.spoof .cls{color:var(--spoof)}.det.mixed .cls{color:var(--mixed)}.det .cls.u{color:var(--unres)}
.det .sev{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--txt)}
.det .reg{font-size:11.5px;margin-top:2px}.det .meta{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:3px}
.insp .ph{font-size:11.5px;color:var(--faint);font-style:italic}
.chip{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.07em;padding:3px 8px;border-radius:4px}
.c-jam{color:#ffd2d8;background:rgba(230,57,80,.18);border:1px solid rgba(230,57,80,.4)}
.c-spoof{color:#ecd6ff;background:rgba(177,77,255,.16);border:1px solid rgba(177,77,255,.4)}
.c-mixed{color:#ffe0c2;background:rgba(255,138,61,.16);border:1px solid rgba(255,138,61,.4)}
.c-elev{color:#ffe6bf;background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.4)}
.c-clean{color:#c4f5ee;background:rgba(31,184,166,.14);border:1px solid rgba(31,184,166,.4)}
.c-unres{color:#ffd9c2;background:rgba(207,122,74,.16);border:1px solid rgba(207,122,74,.4)}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}
.ig div{background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:6px 8px}
.ig .k{font-size:9.5px;color:var(--muted)}.ig .v{font-family:var(--mono);font-size:14px;font-weight:600;margin-top:1px}
.note{font-size:11px;line-height:1.5;color:var(--muted)}.note b{color:var(--txt)}
.illusnote{font-size:10.5px;color:#ffd9a0;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
border-radius:4px;padding:7px 9px;margin-top:8px;line-height:1.45}
.legend{position:absolute;left:12px;bottom:12px;z-index:500;background:rgba(12,24,37,.86);
border:1px solid var(--line);border-radius:6px;padding:9px 11px;font-family:var(--mono);font-size:10px}
.legend .row{display:flex;align-items:center;gap:7px;margin:3px 0;color:var(--muted)}
.sw{width:13px;height:13px;border-radius:3px}
@media(max-width:840px){#app{grid-template-columns:1fr;grid-template-rows:auto 50vh 1fr}
#map{grid-row:2}aside{grid-column:1;grid-row:3;border-left:none;border-top:1px solid var(--line)}}
</style></head><body>
<div id="app">
<header><div class="brand"><b>GNSS INTERFERENCE — JAMMING vs SPOOFING</b>
<span>Layer 1: real interference (GPSJam) · Layer 2: classification (classify.py)</span></div>
__BADGE__</header>
<div id="map"><div class="legend">
<div class="row"><span class="sw" style="background:var(--jam)"></span>Jamming</div>
<div class="row"><span class="sw" style="background:var(--spoof)"></span>Spoofing</div>
<div class="row"><span class="sw" style="background:var(--unres)"></span>Degraded (unresolved)</div>
<div class="row"><span class="sw" style="background:var(--elev)"></span>Elevated</div>
<div class="row"><span class="sw" style="background:var(--clean)"></span>Clean</div></div></div>
<aside>
<div class="sec"><h2>Theater · __DATE__</h2><div class="stats">
<div class="stat jam"><div class="n" id="n-jam">0</div><div class="l">Jamming cells</div></div>
<div class="stat spoof"><div class="n" id="n-spoof">0</div><div class="l">Spoofing cells</div></div>
<div class="stat elev"><div class="n">__ELEV__</div><div class="l">Elevated cells</div></div>
<div class="stat clean"><div class="n">__AC__</div><div class="l">Aircraft sampled</div></div></div>
__ILLUS__</div>
<div class="sec insp" id="insp"><h2>Cell inspector</h2><div class="ph">Tap any cell to see its classification and real aircraft counts.</div></div>
<div class="sec"><h2>Classified hotspots</h2><div class="feed" id="feed"></div></div>
</aside></div>
<script>
const DATA=__GEOJSON__, LBL=__LABELS__;
const map=L.map('map',{attributionControl:false,minZoom:3,maxZoom:7}).setView([41,33],4);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{subdomains:'abcd'}).addTo(map);
const rend=L.canvas({padding:.5});
const C={clean:'#1fb8a6',elev:'#f59e0b',JAMMING:'#e63950',SPOOFING:'#b14dff','MIXED':'#ff8a3d'};
const UNRES='#cf7a4a';
function key(p){return Math.round(p.lat*100)+'_'+Math.round(p.lon*100);}
function clsOf(p){return p.band==='high'?(LBL[key(p)]||'DEGRADED (unresolved)'):null;}
function style(p){
 if(p.band==='clean')return{renderer:rend,color:'#163029',weight:.4,fillColor:C.clean,fillOpacity:.10};
 if(p.band==='elev')return{renderer:rend,color:C.elev,weight:1,fillColor:C.elev,fillOpacity:.32};
 const cls=clsOf(p);const col=C[cls]||UNRES;
 return{renderer:rend,color:col,weight:1,fillColor:col,fillOpacity:Math.min(.82,.45+p.s*.6)};}
let nJam=0,nSpoof=0;
const layer=L.geoJSON(DATA,{style:f=>style(f.properties),
 onEachFeature:(f,l)=>{const c=clsOf(f.properties);if(c==='JAMMING')nJam++;else if(c==='SPOOFING')nSpoof++;
  l.on('click',()=>inspect(f.properties));}}).addTo(map);
document.getElementById('n-jam').textContent=nJam;
document.getElementById('n-spoof').textContent=nSpoof;

function chip(cls){
 if(cls==='JAMMING')return['c-jam','JAMMING'];if(cls==='SPOOFING')return['c-spoof','SPOOFING'];
 if(cls==='MIXED')return['c-mixed','MIXED'];return['c-unres','DEGRADED · UNRESOLVED'];}
function inspect(p){
 const cls=clsOf(p);
 let head,body;
 if(cls){const ch=chip(cls);head=`<span class="chip ${ch[0]}">${ch[1]}</span>`;
  body=cls==='JAMMING'?'Classified jamming: aircraft positions went stale while planes stayed in contact — receivers denied a fix.':
       cls==='SPOOFING'?'Classified spoofing: positions kept updating but tracks became impossible (teleports / a fake-location cluster).':
       cls==='MIXED'?'Both signatures present in this cell.':
       'Interference is real but the movement signature was ambiguous — needs more aircraft samples.';}
 else{const ch=p.band==='elev'?['c-elev','ELEVATED']:['c-clean','CLEAN'];head=`<span class="chip ${ch[0]}">${ch[1]}</span>`;
  body=p.band==='elev'?'Degradation above background but below the confident threshold.':'Clean, consistent GPS. No interference signature.';}
 document.getElementById('insp').innerHTML=`<h2>Cell inspector</h2>
 <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">${head}
 <span style="font-family:var(--mono);font-size:11px;color:var(--muted)">${p.lat}°N ${p.lon}°E</span></div>
 <div style="font-size:12px">${p.reg!=='—'?'Near '+p.reg:'Open airspace'}</div>
 <div class="ig"><div><div class="k">Aircraft sampled</div><div class="v">${p.g+p.b}</div></div>
 <div><div class="k">Interference</div><div class="v">${Math.round(p.s*100)}%</div></div>
 <div><div class="k">Good fix</div><div class="v" style="color:var(--clean)">${p.g}</div></div>
 <div><div class="k">Degraded</div><div class="v" style="color:var(--jam)">${p.b}</div></div></div>
 <div class="note">${body}</div>`;}

const feed=document.getElementById('feed');
DATA.hotspots.forEach(h=>{const k=Math.round(h.lat*100)+'_'+Math.round(h.lon*100);
 const cls=LBL[k]||'DEGRADED (unresolved)';
 const cn=cls==='JAMMING'?'jam':cls==='SPOOFING'?'spoof':cls==='MIXED'?'mixed':'';
 const cu=cn?'':' u';
 const d=document.createElement('div');d.className='det '+cn;
 d.innerHTML=`<div class="top"><span class="cls${cu}">${cls.replace(' (unresolved)','')}</span><span class="sev">${Math.round(h.s*100)}%</span></div>
 <div class="reg">${h.reg!=='—'?h.reg:'Open airspace'}</div>
 <div class="meta">${h.b} of ${h.g+h.b} aircraft degraded · ${h.lat}N ${h.lon}E</div>`;
 d.onclick=()=>map.setView([h.lat,h.lon],6);feed.appendChild(d);});
</script></body></html>'''

illus = ('<div class="illusnote"><b>Illustrative colors.</b> Jamming/spoofing here is assigned from each '
 'region\'s <i>documented</i> dominant interference type, not live movement analysis. Run '
 '<code>classify.py</code> with OpenSky to replace these with real per-cell labels.</div>') if prov=="illustrative" else ""

html=(html.replace("__GEOJSON__",gj).replace("__LABELS__",labels_js)
 .replace("__BADGE__",badge).replace("__ILLUS__",illus)
 .replace("__DATE__",data["date"]).replace("__ELEV__",str(st["elev"]))
 .replace("__AC__",f"{st['aircraft']:,}"))
open("gnss-jamming-spoofing-map.html","w").write(html)
print("wrote classified map, KB:",round(len(html)/1024),"| prov:",prov)

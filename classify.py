#!/usr/bin/env python3
"""
LAYER 2 — jamming vs spoofing classifier.

Layer 1 (detect.py) tells you *where* GPS is degraded. It cannot tell you
*which kind* of attack, because jamming and spoofing both show up as degraded
navigation integrity. Layer 2 resolves that by looking at how aircraft are
actually moving, using free OpenSky position data.

THE SIGNATURES (this is the whole idea)
  • Jamming  — the receiver is denied a fix. Aircraft keep talking (Mode S
               still gets through) but their *position stops updating*: the gap
               between "last contact" and "last position update" blows out, and
               fewer aircraft report position at all. Movement, where present,
               is normal. No teleports.
  • Spoofing — the receiver is fed a confident *false* fix. Position keeps
               updating (low staleness) but the track becomes impossible:
               aircraft teleport between reports, or many unrelated aircraft
               pile onto one fake point (the classic "spoof magnet" over an
               airport). 

So: high interference + stale/absent positions  → JAMMING
    high interference + teleports / fake cluster → SPOOFING
    both                                          → MIXED

DATA
  OpenSky state vectors (free, non-commercial; OAuth2 client-credentials since
  Mar 2026). Set OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in your environment.
  OpenSky gives position, velocity, last_contact and time_position per aircraft
  — everything the signatures above need. (It does NOT give NIC/NACp; that's the
  deeper pyModeS path, noted in the README.)

  Heads-up: the free tier is rate-limited by daily credits. Poll a tight
  bounding box at a modest interval, not the whole sky every few seconds.

Usage:
    pip install requests h3
    export OPENSKY_CLIENT_ID=...    OPENSKY_CLIENT_SECRET=...
    python classify.py                 # classify hot cells from theater.geojson
    python classify.py --region eastern-med   # classify one box live (no Layer 1)
    python classify.py --bbox 32,37,31,37     # ...or a custom box lamin,lamax,lomin,lomax
    python classify.py --selftest      # prove the logic on known fixtures (no network)
    python classify.py --demo          # illustrative region-based labels (no network)

Region presets: eastern-med, black-sea, baltic, gulf, moscow
Optional: --polls N --interval SEC  (default 6 × 15s)
"""
import json, math, os, sys, time

# ---- thresholds (tunable; documented so reviewers see the reasoning) --------
ELEV               = 0.10     # below this interference, a cell is treated as clean
IMPOSSIBLE_MPS     = 500      # ~1800 km/h; no civil airliner moves this fast
TELEPORT_KM        = 150      # a single >150 km step between reports is non-physical
STALE_SECONDS      = 30       # position not updating for >30 s while still in contact
CLUSTER_RADIUS_KM  = 3        # distinct enroute aircraft within this radius = "magnet"
CLUSTER_MIN        = 5        # how many piled-up aircraft make a spoof cluster
JUMP_RATE_HI       = 0.15     # share of aircraft teleporting that flags spoofing
STALE_RATE_HI      = 0.35     # share of aircraft with stale position that flags jamming


def haversine_km(a, b):
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0]); dl = math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(h)))


# ============================================================================
#  CORE LOGIC  (pure functions — these are what --selftest exercises)
# ============================================================================
def aircraft_features(track):
    """track = list of samples sorted by time:
         {t, lat, lon, last_contact, time_position, on_ground}
       Returns this aircraft's (teleported, stale) booleans."""
    teleported = False
    for s in track:
        if s["last_contact"] is not None and s["time_position"] is not None:
            if (s["last_contact"] - s["time_position"]) > STALE_SECONDS:
                stale = True
                break
    else:
        stale = False
    pts = [s for s in track if s["lat"] is not None]
    for prev, cur in zip(pts, pts[1:]):
        dt = max(1.0, cur["t"] - prev["t"])
        dist = haversine_km((prev["lat"], prev["lon"]), (cur["lat"], cur["lon"]))
        if dist > TELEPORT_KM or (dist * 1000 / dt) > IMPOSSIBLE_MPS:
            teleported = True
            break
    return teleported, stale


def cluster_score(tracks):
    """Largest group of *distinct, airborne* aircraft sharing one tiny location —
    the spoof-magnet signature. Returns (count, center)."""
    latest = []
    for icao, tr in tracks.items():
        pts = [s for s in tr if s["lat"] is not None and not s.get("on_ground")]
        if pts:
            last = pts[-1]
            latest.append((icao, last["lat"], last["lon"]))
    best, center = 0, None
    for i, (_, la, lo) in enumerate(latest):
        group = [o for o in latest if haversine_km((la, lo), (o[1], o[2])) <= CLUSTER_RADIUS_KM]
        if len(group) > best:
            best, center = len(group), (la, lo)
    return best, center


def classify_cell(interference, tracks):
    """tracks = {icao24: [samples...]}. Returns label + confidence + reasons.
    interference: GPSJam score for the cell, or None to judge purely on movement
    (used by --region, which runs without Layer 1)."""
    n = len(tracks)
    if not n:
        return {"label": "NO DATA", "confidence": 0.3, "features": {"n": 0},
                "reasons": ["No aircraft reporting positions in this box — empty airspace, "
                            "or jamming/closure suppressing reports (itself a signal)."]}
    if interference is not None and interference < ELEV:
        return {"label": "CLEAN", "confidence": 0.9, "reasons":
                ["Aircraft report consistent navigation integrity; no interference."],
                "features": {"n": n}}

    teleporters = stales = 0
    for tr in tracks.values():
        tp, st = aircraft_features(sorted(tr, key=lambda s: s["t"]))
        teleporters += tp; stales += st
    jump_rate  = teleporters / n if n else 0
    stale_rate = stales / n if n else 0
    clust, center = cluster_score(tracks)

    spoof_evidence = (jump_rate >= JUMP_RATE_HI) or (clust >= CLUSTER_MIN)
    jam_evidence   = (stale_rate >= STALE_RATE_HI) and jump_rate < JUMP_RATE_HI

    reasons, feats = [], {"n": n, "jump_rate": round(jump_rate, 2),
                          "stale_rate": round(stale_rate, 2), "cluster": clust}
    if spoof_evidence and jam_evidence:
        label, conf = "MIXED", 0.6
        reasons.append(f"Both signatures present: {int(jump_rate*100)}% teleporting AND "
                       f"{int(stale_rate*100)}% with stale positions.")
    elif spoof_evidence:
        label = "SPOOFING"
        conf = min(0.95, 0.5 + jump_rate + 0.08 * clust)
        if clust >= CLUSTER_MIN:
            reasons.append(f"{clust} airborne aircraft clustered within {CLUSTER_RADIUS_KM} km "
                           f"of one point {center} — a spoof 'magnet'.")
        if jump_rate >= JUMP_RATE_HI:
            reasons.append(f"{int(jump_rate*100)}% of aircraft reported physically impossible "
                           f"position jumps while still claiming a fix.")
        reasons.append("Positions keep updating but are false → spoofing, not jamming.")
    elif jam_evidence:
        label, conf = "JAMMING", min(0.95, 0.5 + stale_rate)
        reasons.append(f"{int(stale_rate*100)}% of aircraft stopped updating position while "
                       f"still in Mode-S contact — receivers denied a fix.")
        reasons.append("No teleports or fake clusters → jamming, not spoofing.")
    else:
        if interference is None:
            label, conf = "CLEAN", min(0.85, 0.5 + 0.03 * n)
            reasons.append("No jamming or spoofing signature in aircraft movement here.")
        else:
            label, conf = "DEGRADED (unresolved)", 0.4
            reasons.append("Interference present but the movement signature is ambiguous "
                           "(too few aircraft, or mixed behavior). Needs more samples.")
    if n < 6:
        conf = min(conf, 0.45)
        reasons.append(f"⚠ Low confidence: only {n} aircraft sampled.")
    return {"label": label, "confidence": round(conf, 2), "reasons": reasons, "features": feats}


# ============================================================================
#  LIVE PATH  (OpenSky)
# ============================================================================
def opensky_token():
    import requests
    cid, sec = os.environ.get("OPENSKY_CLIENT_ID"), os.environ.get("OPENSKY_CLIENT_SECRET")
    if not cid or not sec:
        raise SystemExit("Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET (create an API "
                         "client at opensky-network.org → Account).")
    url = ("https://auth.opensky-network.org/auth/realms/opensky-network/"
           "protocol/openid-connect/token")
    r = requests.post(url, data={"grant_type": "client_credentials",
                                 "client_id": cid, "client_secret": sec}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def poll_states(bbox, token):
    import requests
    lamin, lamax, lomin, lomax = bbox
    r = requests.get("https://opensky-network.org/api/states/all",
                     params={"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax},
                     headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json().get("states") or []


def collect_tracks(bbox, polls=6, interval=15, token=None):
    """Snapshot the box several times and stitch per-aircraft tracks by icao24."""
    tracks = {}
    for k in range(polls):
        for s in poll_states(bbox, token):
            icao = s[0]
            tracks.setdefault(icao, []).append(
                {"t": s[4] or time.time(), "lat": s[6], "lon": s[5],
                 "time_position": s[3], "last_contact": s[4], "on_ground": s[8]})
        if k < polls - 1:
            time.sleep(interval)
    return tracks


REGIONS_BBOX = {                       # (lamin, lamax, lomin, lomax)
    "eastern-med": (32.0, 37.0, 31.0, 37.0),
    "black-sea":   (41.0, 47.0, 30.0, 42.0),
    "baltic":      (53.0, 57.5, 18.0, 24.0),
    "gulf":        (24.0, 28.5, 53.0, 58.0),
    "moscow":      (54.0, 58.5, 33.0, 40.0),
}


def _opt(args, flag, default):
    return args[args.index(flag) + 1] if flag in args else default


def run_region(args):
    """Classify one bounding box live from OpenSky, with no Layer 1 needed.
    Movement alone decides clean / jamming / spoofing for the box."""
    if "--bbox" in args:
        bbox = tuple(float(x) for x in _opt(args, "--bbox", "").split(","))
        name = "custom"
    else:
        name = _opt(args, "--region", "")
        if name not in REGIONS_BBOX:
            raise SystemExit("Pick --region one of: " + ", ".join(REGIONS_BBOX)
                             + "   or pass --bbox lamin,lamax,lomin,lomax")
        bbox = REGIONS_BBOX[name]
    polls = int(_opt(args, "--polls", 6))
    interval = int(_opt(args, "--interval", 15))

    token = opensky_token()
    print(f"Watching '{name}' {bbox} — {polls} polls × {interval}s "
          f"(~{polls*interval}s of position history)...")
    tracks = collect_tracks(bbox, polls=polls, interval=interval, token=token)
    res = classify_cell(None, tracks)          # None = judge purely on movement
    print(f"\n  {name}: {res['label']}   (confidence {res['confidence']})")
    print(f"  features: {res['features']}")
    for r in res["reasons"]:
        print(f"    · {r}")


def run_live():
    if not os.path.exists("theater.geojson"):
        raise SystemExit("Run detect.py first to produce theater.geojson (the hot cells).")
    data = json.load(open("theater.geojson"))
    hot = [f for f in data["features"] if f["properties"]["band"] == "high"]
    if not hot:
        print("No high-interference cells in this day; nothing to classify."); return
    token = opensky_token()
    # group hot cells into a few bounding boxes to respect rate limits
    print(f"Classifying {len(hot)} hot cells via OpenSky "
          f"(polling positions; this takes ~1–2 min per region)...\n")
    labels = []
    for f in hot[:8]:                      # cap for the demo; widen as credits allow
        la, lo = f["properties"]["lat"], f["properties"]["lon"]
        bbox = (la - 0.6, la + 0.6, lo - 0.6, lo + 0.6)
        tracks = collect_tracks(bbox, token=token)
        res = classify_cell(f["properties"]["s"], tracks)
        labels.append({"lat": la, "lon": lo, "reg": f["properties"]["reg"], **res})
        print(f"  {la:.1f}N {lo:.1f}E  {f['properties']['reg']:<24} → "
              f"{res['label']:<22} (conf {res['confidence']})")
        for r in res["reasons"]:
            print(f"        · {r}")
    json.dump({"provenance": "live", "source": "OpenSky position tracks",
               "labels": labels}, open("labels.json", "w"), indent=2)
    print(f"\nWrote labels.json ({len(labels)} cells, live).")


# ============================================================================
#  DEMO LABELS  (illustrative only — region-based, NOT live movement analysis)
# ============================================================================
# Dominant interference type documented for each region across 2023–2026.
# Eastern Med has been a persistent spoofing theater; the Black Sea, Baltic and
# Russian interior are predominantly jamming. This lets the map show distinct
# colors before you run the live classifier — it is clearly badged "illustrative".
DOC_TYPE = {
    "Eastern Mediterranean": "SPOOFING", "Nile Delta & Sinai": "SPOOFING",
    "Black Sea": "JAMMING", "Kaliningrad & Baltic": "JAMMING",
    "Moscow region": "JAMMING", "Strait of Hormuz / Gulf": "JAMMING",
}


def demo_labels():
    if not os.path.exists("theater.geojson"):
        raise SystemExit("Run detect.py first to produce theater.geojson.")
    data = json.load(open("theater.geojson"))
    hot = [f for f in data["features"] if f["properties"]["band"] == "high"]
    out = []
    for f in hot:
        p = f["properties"]
        out.append({"lat": p["lat"], "lon": p["lon"], "reg": p["reg"],
                    "label": DOC_TYPE.get(p["reg"], "DEGRADED (unresolved)"),
                    "confidence": None,
                    "reasons": ["Illustrative: based on the documented dominant "
                                "interference type for this region, not live "
                                "per-aircraft movement analysis."]})
    json.dump({"provenance": "illustrative",
               "source": "documented regional patterns (placeholder for live OpenSky run)",
               "labels": out}, open("labels.json", "w"), indent=2)
    print(f"Wrote {len(out)} ILLUSTRATIVE labels to labels.json (region-based).")
    print("Replace with real per-cell labels:  python classify.py   (needs OpenSky)")


# ============================================================================
#  SELF-TEST  (proves the classifier on known fixtures, no network)
# ============================================================================
def selftest():
    def smooth(icao, lat, lon, fresh=True):
        # normal enroute aircraft: ~0.03 deg (~3 km) per 15 s ≈ 220 m/s, position fresh
        return [{"t": 1000 + 15*i, "lat": lat + 0.03*i, "lon": lon + 0.03*i,
                 "time_position": 1000 + 15*i, "last_contact": 1000 + 15*i,
                 "on_ground": False} for i in range(4)]

    # 1) CLEAN — low interference, normal motion
    clean = {f"AC{i}": smooth(i, 48 + i*0.1, 8) for i in range(12)}

    # 2) JAMMING — high interference, positions go STALE (last_contact >> time_position),
    #    no teleports, no cluster
    jam = {}
    for i in range(10):
        tr = smooth(i, 44 + i*0.05, 36)
        for s in tr:               # freeze position updates 120 s behind contact
            s["time_position"] = s["last_contact"] - 120
        jam[f"JM{i}"] = tr

    # 3) SPOOFING — high interference, teleports + 7 aircraft on one fake point,
    #    positions fresh
    spoof = {}
    for i in range(8):             # teleporters
        spoof[f"SP{i}"] = [
            {"t": 1000, "lat": 34.0, "lon": 34.0, "time_position": 1000, "last_contact": 1000, "on_ground": False},
            {"t": 1015, "lat": 34.0, "lon": 35.8, "time_position": 1015, "last_contact": 1015, "on_ground": False}]
    for i in range(7):             # spoof magnet: all parked on one coordinate
        spoof[f"MG{i}"] = [{"t": 1000, "lat": 33.50, "lon": 33.50,
                            "time_position": 1000, "last_contact": 1000, "on_ground": False}]

    cases = [("CLEAN", 0.03, clean, "CLEAN"),
             ("JAMMING", 0.45, jam, "JAMMING"),
             ("SPOOFING", 0.45, spoof, "SPOOFING")]
    print("Layer-2 classifier self-test\n" + "-"*52)
    ok = True
    for name, interf, tracks, expect in cases:
        res = classify_cell(interf, tracks)
        passed = res["label"] == expect
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<9} expected {expect:<9} "
              f"got {res['label']:<22} conf {res['confidence']}")
        print(f"        features: {res['features']}")
        print(f"        {res['reasons'][0]}")
    print("-"*52)
    print("ALL PASS ✓" if ok else "SOME FAILED ✗")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--demo" in sys.argv:
        demo_labels()
        sys.exit(0)
    if "--region" in sys.argv or "--bbox" in sys.argv:
        run_region(sys.argv[1:])
        sys.exit(0)
    run_live()

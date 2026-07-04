#!/usr/bin/env python3
"""
GNSS interference detector — live pipeline.

Pulls one day of real aircraft GPS-integrity data from GPSJam (gpsjam.org),
scores every hex cell for interference using GPSJam's own formula, and writes
a theater.geojson the map front-end loads.

Data: GPSJam, CC-BY, John Wiseman / ADS-B Exchange.
Each daily file is one CSV:  hex, count_good_aircraft, count_bad_aircraft
where `hex` is an H3 resolution-4 cell index.

WHY FETCHING LIVES HERE AND NOT IN THE BROWSER
The map is a static page; a browser can't pull GPSJam's files directly
(no CORS headers + sandbox). So the fetch runs here, server-side, and writes
a data file the page reads. That separation is the correct architecture and
the thing to be able to explain in an interview.

Usage:
    pip install requests h3
    python detect.py                      # most recent available day (auto)
    python detect.py 2023-10-10           # a specific date
    python detect.py --local path.csv     # score a CSV you already have
"""
import csv, json, math, sys, datetime
import h3

# ---- Theater of interest (Europe / Med / Black Sea / Baltic / Gulf) ----------
LATMIN, LATMAX, LONMIN, LONMAX = 20, 60, 8, 62

# Documented hotspots — used only to attach a human-readable label to a cell.
REGIONS = [
    ("Eastern Mediterranean", 34.0, 34.6), ("Nile Delta & Sinai", 30.6, 33.2),
    ("Kaliningrad & Baltic", 54.8, 20.6),  ("Black Sea", 44.2, 36.5),
    ("Strait of Hormuz / Gulf", 26.7, 56.3), ("Moscow region", 55.6, 37.6),
]

HIGH, ELEV = 0.30, 0.10   # interference-score thresholds (red / amber)
DATA_URL = "https://gpsjam.org/data/{date}-h3_4.csv"
MANIFEST_URL = "https://gpsjam.org/data/manifest.csv"
UA = {"User-Agent": "gnss-detector/1.0"}


def latest_available():
    """Ask GPSJam's manifest for the newest published day. The last row is the
    freshest date; `suspect=true` means that day's data may be incomplete."""
    import requests
    r = requests.get(MANIFEST_URL, headers=UA, timeout=30)
    r.raise_for_status()
    rows = [row for row in csv.DictReader(r.text.splitlines()) if row.get("date")]
    last = rows[-1]
    return last["date"], last.get("suspect", "").strip().lower() == "true"


def nearest_region(lat, lon):
    best, bd = "—", 1e9
    for name, la, lo in REGIONS:
        d = math.hypot(lat - la, (lon - lo) * math.cos(math.radians(lat)))
        if d < bd:
            bd, best = d, name
    return best if bd < 6 else "—"


def score(good, bad):
    """GPSJam's exact metric. The -1 suppresses false alarms in thin cells."""
    tot = good + bad
    return (max(0, bad - 1) / tot) if tot else 0.0


def fetch_csv(date):
    import requests
    url = DATA_URL.format(date=date)
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code == 404:
        raise SystemExit(f"No data posted for {date} yet (GPSJam updates after ~00:00 UTC). "
                         f"Try an earlier date.")
    r.raise_for_status()
    return r.text.splitlines()


def main():
    args = sys.argv[1:]
    suspect = False
    if args and args[0] == "--local":
        date = "local"
        lines = open(args[1]).read().splitlines()
    elif args:
        date = args[0]
        print(f"Fetching GPSJam data for {date} ...")
        lines = fetch_csv(date)
    else:
        date, suspect = latest_available()
        print(f"Most recent available day: {date}"
              + ("  [⚠ flagged possibly-incomplete]" if suspect else ""))
        lines = fetch_csv(date)

    feats, hot = [], []
    nclean = nelev = nhigh = ac = 0
    for row in csv.DictReader(lines):
        try:
            good = int(row["count_good_aircraft"]); bad = int(row["count_bad_aircraft"])
        except (KeyError, ValueError):
            continue
        tot = good + bad
        if tot < 4:
            continue
        lat, lon = h3.cell_to_latlng(row["hex"])
        if not (LATMIN <= lat <= LATMAX and LONMIN <= lon <= LONMAX):
            continue
        s = score(good, bad)
        if s >= HIGH:   band = "high"; nhigh += 1
        elif s >= ELEV: band = "elev"; nelev += 1
        else:
            band = "clean"; nclean += 1
            if tot < 110:        # keep only well-sampled clean cells as context
                continue
        ac += tot
        ring = [[round(lo, 2), round(la, 2)] for la, lo in h3.cell_to_boundary(row["hex"])]
        ring.append(ring[0])
        reg = nearest_region(lat, lon)
        feats.append({"type": "Feature",
                      "properties": {"g": good, "b": bad, "s": round(s, 3), "band": band,
                                     "lat": round(lat, 2), "lon": round(lon, 2), "reg": reg},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
        if s >= HIGH and tot >= 6:
            hot.append((s, good, bad, round(lat, 2), round(lon, 2), reg))

    hot.sort(reverse=True)
    out = {"type": "FeatureCollection", "date": date,
           "source": "GPSJam (gpsjam.org) · CC-BY · John Wiseman / ADS-B Exchange",
           "stats": {"clean": nclean, "elev": nelev, "high": nhigh, "aircraft": ac, "cells": len(feats)},
           "hotspots": [{"s": round(s, 3), "g": g, "b": b, "lat": la, "lon": lo, "reg": r}
                        for s, g, b, la, lo, r in hot[:16]],
           "features": feats}
    json.dump(out, open("theater.geojson", "w"), separators=(",", ":"))

    print(f"\n  {date} · {len(feats)} cells · {ac:,} aircraft sampled"
          + ("   [⚠ day flagged possibly-incomplete]" if suspect else ""))
    print(f"  HIGH interference: {nhigh}   elevated: {nelev}")
    print("\n  Top interference cells (real):")
    for s, g, b, la, lo, r in hot[:10]:
        print(f"    {int(s*100):>3}%  {b:>3}/{g+b:<4} aircraft degraded  {la:>5}N {lo:>5}E  {r}")
    print("\n  Wrote theater.geojson  →  open the map to view.")
    print("  NB: this layer shows interference *intensity* only. Jamming vs spoofing")
    print("      is Layer 2 — it needs message-level position data (pyModeS / ADS-B Exchange).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One-time migration: import legacy CSV vitals into SQLite.

Assumes CSV has at least:
- patient_id, timestamp, sensor, value
(Extra columns are ignored.)

Understands BP strings like "120/80".
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from edge_core.config import ProductionConfig
from edge_core.storage import init_sqlite_if_needed, get_conn

# Canonicalization map (keep aligned with DataManager)
CANON = {
    "ECG": "ECG", "SpO2": "SpO2", "BP_SYS": "BP_SYS", "BP_DIA": "BP_DIA", "TEMP": "TEMP",
    "heart_rate": "ECG", "oxygen_saturation": "SpO2",
    "bp_systolic": "BP_SYS", "bp_diastolic": "BP_DIA",
    "temperature": "TEMP", "temp": "TEMP",
    "bp": "BP", "blood_pressure": "BP",
}
def _canon(name: str) -> str:
    if not name: return ""
    k = str(name).strip()
    return CANON.get(k, CANON.get(k.upper(), k.upper()))

def migrate(csv_path: str, sqlite_path: str, dry_run: bool = False) -> int:
    csv = Path(csv_path)
    if not csv.exists():
        print(f"[!] CSV not found: {csv}")
        return 0

    # Ensure DB exists
    init_sqlite_if_needed(sqlite_path)

    df = pd.read_csv(csv)
    needed = ["patient_id", "timestamp", "sensor", "value"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    # Normalize timestamps to "YYYY-MM-DD HH:MM:SS"
    def _norm_ts(x):
        if pd.isna(x): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df["timestamp"] = df["timestamp"].map(_norm_ts)

    rows = []
    bp_split = 0
    skipped = 0

    for _, r in df.iterrows():
        pid = r["patient_id"]
        ts  = r["timestamp"]
        sen = _canon(r["sensor"])
        val = r["value"]
        unit = r["unit"] if "unit" in df.columns else ""

        if not pid or not ts or not sen:
            skipped += 1
            continue

        # BP "120/80" split
        if isinstance(val, str) and sen in ("BP", "BLOOD_PRESSURE") and "/" in val:
            try:
                s, d = val.split("/")
                rows.append((pid, ts, "BP_SYS", float(s.strip()), unit, "csv_import"))
                rows.append((pid, ts, "BP_DIA", float(d.strip()), unit, "csv_import"))
                bp_split += 1
                continue
            except Exception:
                skipped += 1
                continue

        # Normal numeric
        try:
            fval = float(val)
        except Exception:
            skipped += 1
            continue
        rows.append((pid, ts, sen, fval, unit, "csv_import"))

    if dry_run:
        print(f"[dry-run] Would insert {len(rows)} rows "
              f"(split {bp_split} BP rows, skipped {skipped}).")
        return len(rows)

    with get_conn(sqlite_path) as conn:
        conn.executemany(
            """INSERT INTO vitals(patient_id, ts, sensor, value, unit, source)
               VALUES (?,?,?,?,?,?)""",
            rows,
        )

    print(f"[ok] Inserted {len(rows)} rows into {sqlite_path} "
          f"(split {bp_split} BP rows, skipped {skipped}).")
    return len(rows)

def main():
    ap = argparse.ArgumentParser(description="Migrate legacy vitals CSV into SQLite.")
    ap.add_argument("--csv", required=True, help="Path to legacy CSV (e.g., data/vitals.csv)")
    ap.add_argument("--db", default=None, help="Path to SQLite DB (default: from settings.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Validate without inserting")
    args = ap.parse_args()

    cfg = ProductionConfig.from_settings()
    sqlite_path = args.db or cfg.sqlite_path
    migrate(args.csv, sqlite_path, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

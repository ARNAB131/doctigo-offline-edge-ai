# edge_core/data_manager.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
from .storage import get_conn  # uses the WAL/PRAGMA setup from Step 3

# Canonical sensor names used across the app/DB
CANON = {
    "ECG": "ECG",
    "SpO2": "SpO2",
    "BP_SYS": "BP_SYS",
    "BP_DIA": "BP_DIA",
    "TEMP": "TEMP",
    # common aliases -> canonical
    "heart_rate": "ECG",
    "oxygen_saturation": "SpO2",
    "bp_systolic": "BP_SYS",
    "bp_diastolic": "BP_DIA",
    "temperature": "TEMP",
    "temp": "TEMP",
}

def _canon(name: str) -> str:
    if not name:
        return ""
    key = str(name).strip()
    return CANON.get(key, CANON.get(key.upper(), key.upper()))

class DataManager:
    """SQLite-only, offline-first data manager."""

    def __init__(self, config):
        # We now use SQLite instead of CSV
        self.db_path = getattr(config, "sqlite_path", "data/edge.db")

    # ---------- Writes ----------

    def store_vital_sign(self, vital: Dict[str, Any] | Any) -> None:
        """
        Accepts either a dict or an object with attrs:
          patient_id, timestamp, sensor_type (or sensor), value, [unit]
        Persists one row in the vitals table.
        """
        pid = getattr(vital, "patient_id", None) or (isinstance(vital, dict) and vital.get("patient_id"))
        ts  = getattr(vital, "timestamp", None)   or (isinstance(vital, dict) and vital.get("timestamp"))
        sen = getattr(vital, "sensor_type", None) or getattr(vital, "sensor", None)
        if sen is None and isinstance(vital, dict):
            sen = vital.get("sensor_type") or vital.get("sensor")
        val = getattr(vital, "value", None)       or (isinstance(vital, dict) and vital.get("value"))
        unit = getattr(vital, "unit", "")         or (isinstance(vital, dict) and vital.get("unit")) or ""

        if not ts:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")

        sensor = _canon(sen)

        # Special case: BP in "120/80" → split into two rows
        if isinstance(val, str) and sensor in ("BP", "BLOOD_PRESSURE") and "/" in val:
            try:
                s, d = val.split("/")
                s = float(s.strip()); d = float(d.strip())
                with get_conn(self.db_path) as conn:
                    conn.execute("""INSERT INTO vitals(patient_id, ts, sensor, value, unit, source)
                                    VALUES (?,?,?,?,?,?)""", (pid, ts, "BP_SYS", s, unit, "edge"))
                    conn.execute("""INSERT INTO vitals(patient_id, ts, sensor, value, unit, source)
                                    VALUES (?,?,?,?,?,?)""", (pid, ts, "BP_DIA", d, unit, "edge"))
                return
            except Exception:
                # fall through to attempt numeric parse below
                pass

        # Normal numeric write
        try:
            fval = float(val)
        except (TypeError, ValueError):
            # ignore malformed values
            return

        with get_conn(self.db_path) as conn:
            conn.execute("""INSERT INTO vitals(patient_id, ts, sensor, value, unit, source)
                            VALUES (?,?,?,?,?,?)""", (pid, ts, sensor, fval, unit, "edge"))

    def bulk_store_vitals(self, rows: Iterable[Dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        payload = []
        for r in rows:
            pid = r.get("patient_id")
            ts  = r.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            sensor = _canon(r.get("sensor") or r.get("sensor_type"))
            try:
                val = float(r.get("value"))
            except (TypeError, ValueError):
                continue
            unit = r.get("unit", "")
            payload.append((pid, ts, sensor, val, unit, "edge"))
        if not payload:
            return 0
        with get_conn(self.db_path) as conn:
            conn.executemany("""INSERT INTO vitals(patient_id, ts, sensor, value, unit, source)
                                VALUES (?,?,?,?,?,?)""", payload)
        return len(payload)

    # ---------- Reads ----------

    def get_patient_vitals_history(
        self, patient_id: str, sensor_type: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        sensor = _canon(sensor_type) if sensor_type else None
        if sensor:
            sql = """SELECT ts, sensor, value, unit
                       FROM vitals WHERE patient_id=? AND sensor=?
                   ORDER BY ts DESC LIMIT ?"""
            params = (patient_id, sensor, limit)
        else:
            sql = """SELECT ts, sensor, value, unit
                       FROM vitals WHERE patient_id=?
                   ORDER BY ts DESC LIMIT ?"""
            params = (patient_id, limit)
        with get_conn(self.db_path) as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        # chronological order
        rows = rows[::-1]
        return [{"timestamp": r[0], "sensor": r[1], "value": r[2], "unit": r[3] or ""} for r in rows]

    def get_multi_sensor_series(
        self, patient_id: str, sensors: Iterable[str], since_ts: Optional[str] = None, limit: int = 500
    ) -> Dict[str, List[Tuple[str, float]]]:
        sens = tuple(_canon(s) for s in sensors)
        q = ["SELECT ts, sensor, value FROM vitals WHERE patient_id=?", f"AND sensor IN ({','.join(['?']*len(sens))})"]
        params: List[Any] = [patient_id, *sens]
        if since_ts:
            q.append("AND ts >= ?")
            params.append(since_ts)
        q.append("ORDER BY ts DESC LIMIT ?")
        params.append(limit)
        sql = " ".join(q)
        out = {s: [] for s in sens}
        with get_conn(self.db_path) as conn:
            for ts, sensor, value in conn.execute(sql, params).fetchall():
                out[sensor].append((ts, value))
        return {k: v[::-1] for k, v in out.items()}  # chronological

    def get_latest_values(self, patient_id: str, sensors: Iterable[str]) -> Dict[str, Optional[float]]:
        sens = tuple(_canon(s) for s in sensors)
        out = {s: None for s in sens}
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                f"""
                SELECT sensor, value FROM (
                    SELECT sensor, value,
                           ROW_NUMBER() OVER (PARTITION BY sensor ORDER BY ts DESC) rn
                      FROM vitals
                     WHERE patient_id=?
                       AND sensor IN ({','.join(['?']*len(sens))})
                ) WHERE rn=1
                """,
                (patient_id, *sens),
            )
            for sensor, value in cur.fetchall():
                out[sensor] = value
        return out

    # ---------- Alerts/Audit/Models passthroughs used by AlertManager ----------
    def create_alert(self, patient_id: str, ts: str, type_: str, severity: str, payload: Dict[str, Any]) -> str:
        import json, uuid
        alert_id = str(uuid.uuid4())
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO alerts(id, patient_id, ts, type, severity, payload) VALUES (?,?,?,?,?,?)",
                (alert_id, patient_id, ts, type_, severity, json.dumps(payload)),
            )
        return alert_id

    def get_alert_statistics(self) -> Dict[str, Any]:
        with get_conn(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            crit  = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity='critical'").fetchone()[0]
        return {"active_alerts": total, "critical": crit}

    # ---------- (Optional) Predictions storage ----------
    def store_prediction(self, patient_id: str, ts: str, payload: Dict[str, Any]) -> None:
        # You can add a 'predictions' table later if you want persistence
        pass

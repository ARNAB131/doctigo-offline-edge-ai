# edge_core/alert_manager.py
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

class AlertManager:
    """
    Offline-first AlertManager:
      • Reads vitals from the provided twin (your current flow) and/or DB fallback.
      • Normalizes sensor names (ecg/ECG, spo2/SpO2, bp_systolic/BP_SYS, etc.).
      • Handles BP as "120/80" or as separate systolic/diastolic readings.
      • Persists every alert via DataManager.create_alert().
      • Stats come from the alerts table (DB-backed).
    """

    # Map many possible input names -> canonical keys we evaluate on
    CANON_MAP: Dict[str, str] = {
        "ecg": "ECG", "heart_rate": "ECG", "hr": "ECG",
        "spo2": "SpO2", "oxygen": "SpO2", "oxygen_saturation": "SpO2",
        "bp_sys": "BP_SYS", "bpsys": "BP_SYS", "bp_systolic": "BP_SYS", "systolic": "BP_SYS",
        "bp_dia": "BP_DIA", "bpdia": "BP_DIA", "bp_diastolic": "BP_DIA", "diastolic": "BP_DIA",
        "bp": "BP", "blood_pressure": "BP",  # special case (e.g., "120/80")
        "temp": "TEMP", "temperature": "TEMP", "body_temperature": "TEMP",
    }

    # Safe + borderline bands (you can later load from rules.yaml)
    RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
        "ECG":   {"safe": (60, 100), "borderline": (50, 110)},
        "SpO2":  {"safe": (95, 100), "borderline": (90, 94)},
        "BP_SYS":{"safe": (90, 120), "borderline": (80, 140)},
        "BP_DIA":{"safe": (60, 80),  "borderline": (50, 95)},
        "TEMP":  {"safe": (36.1, 37.5), "borderline": (35.8, 38.0)},
    }

    def __init__(self, config, data_manager):
        self.config = config
        self.data_manager = data_manager  # must expose create_alert() and get_alert_statistics()

    # ---------- Public API ----------
    def generate_alert(self, patient_id: str, twin: Dict[str, Any] | None, predictions: List[Dict[str, Any]] | None):
        """
        - Reads vitals from twin["vitals"] (list of dicts/objects) if available.
        - If twin has no vitals, fallback to DB latest values.
        - Evaluates against bands; persists alerts into DB.
        - Returns a small message dict for on-screen display (compatible with your current use).
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        alerts_text: List[str] = []

        # 1) Collect latest vitals from twin (preferred path)
        latest: Dict[str, Optional[float]] = {}
        vitals = (twin or {}).get("vitals", [])

        # Parse each vital in twin: accept obj or dict
        for vital in vitals:
            sensor_type = getattr(vital, "sensor_type", getattr(vital, "sensor", None))
            if sensor_type is None and isinstance(vital, dict):
                sensor_type = vital.get("sensor_type") or vital.get("sensor") or vital.get("type")
            value = getattr(vital, "value", None)
            if value is None and isinstance(vital, dict):
                value = vital.get("value")

            if not sensor_type:
                continue
            canon = self._canon(sensor_type)

            # Special case: BP in "120/80"
            if canon == "BP" and isinstance(value, str) and ("/" in value):
                try:
                    sys_val, dia_val = value.split("/")
                    sys_val = float(sys_val.strip())
                    dia_val = float(dia_val.strip())
                    latest["BP_SYS"] = sys_val
                    latest["BP_DIA"] = dia_val
                except Exception:
                    pass
                continue

            # Normal numeric value
            if canon in ("ECG", "SpO2", "BP_SYS", "BP_DIA", "TEMP"):
                try:
                    latest[canon] = float(value)
                except (TypeError, ValueError):
                    continue

        # 2) DB fallback if twin provided nothing for a channel we care about
        missing = [k for k in ("ECG", "SpO2", "BP_SYS", "BP_DIA", "TEMP") if k not in latest]
        if missing:
            db_latest = self.data_manager.get_latest_values(patient_id, missing)
            for k, v in db_latest.items():
                if v is not None:
                    latest[k] = v

        # 3) Evaluate bands and persist alerts
        for sensor, val in latest.items():
            if val is None or sensor not in self.RANGES:
                continue
            ranges = self.RANGES[sensor]
            safe_low, safe_high = ranges["safe"]
            border_low, border_high = ranges["borderline"]

            severity = None
            reason = ""
            if val < border_low or val > border_high:
                severity = "critical"
                reason = f"Value {val} outside borderline [{border_low}, {border_high}]"
            elif val < safe_low or val > safe_high:
                severity = "warn"
                reason = f"Value {val} outside safe [{safe_low}, {safe_high}]"

            if severity:
                alerts_text.append(f"{sensor} out of range: {val}")
                payload = {"sensor": sensor, "value": val, "bands": ranges, "reason": reason}
                self.data_manager.create_alert(
                    patient_id=patient_id, ts=now, type_="band", severity=severity, payload=payload
                )

        # 4) (Optional) Use predictions to add forecast-based alerts
        if predictions:
            for p in predictions:
                # If your predictions are dicts like {"systolic_bp": 151.2, ...}:
                sbp = None
                if isinstance(p, dict):
                    cand = p.get("systolic_bp")
                    if isinstance(cand, (int, float)):  # point forecast
                        sbp = float(cand)
                    elif isinstance(cand, dict):        # quantiles structure
                        sbp = cand.get("p50") or cand.get("point")
                if sbp and sbp > 150:
                    alerts_text.append(f"Forecast systolic high: {sbp}")
                    self.data_manager.create_alert(
                        patient_id=patient_id,
                        ts=now,
                        type_="forecast",
                        severity="warn",
                        payload={"metric": "systolic_bp", "pred": sbp},
                    )

        if alerts_text:
            return {
                "title": "🚨 Alert",
                "message": "\n".join(alerts_text),
            }
        return None

    def get_alert_statistics(self) -> Dict[str, Any]:
        # DB-backed counts
        return self.data_manager.get_alert_statistics()

    # ---------- helpers ----------
    @classmethod
    def _canon(cls, name: str) -> str:
        key = (name or "").strip().lower()
        return cls.CANON_MAP.get(key, key.upper())  # default to upper for unknowns

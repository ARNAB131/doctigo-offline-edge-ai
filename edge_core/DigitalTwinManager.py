# edge_core/digital_twin_manager.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime

class DigitalTwinManager:
    """
    Manages a digital twin per patient.

    Compatible with your existing calls:
      - update_twin(patient_id, vitals, predictions=None)
      - get_twin(patient_id)
      - get_all_twins_summary()

    Improvements:
      • Normalizes vitals (ECG/SpO2/BP_SYS/BP_DIA/TEMP), supports "120/80" BP strings.
      • Falls back to DataManager.get_latest_values(...) if vitals list is empty/partial.
      • Computes a simple risk_score (0..1) from latest vitals + optional predictions.
      • Stores a compact state (latest, predictions, risk_score, updated_ts).
    """

    # Name normalization to canonical keys
    CANON_MAP = {
        "ecg": "ECG", "heart_rate": "ECG", "hr": "ECG",
        "spo2": "SpO2", "oxygen": "SpO2", "oxygen_saturation": "SpO2",
        "bp_sys": "BP_SYS", "bpsys": "BP_SYS", "bp_systolic": "BP_SYS", "systolic": "BP_SYS",
        "bp_dia": "BP_DIA", "bpdia": "BP_DIA", "bp_diastolic": "BP_DIA", "diastolic": "BP_DIA",
        "bp": "BP", "blood_pressure": "BP",
        "temp": "TEMP", "temperature": "TEMP", "body_temperature": "TEMP",
    }

    # Bands (keep aligned with AlertManager for consistency)
    RANGES = {
        "ECG":    {"safe": (60, 100), "borderline": (50, 110)},
        "SpO2":   {"safe": (95, 100), "borderline": (90, 94)},
        "BP_SYS": {"safe": (90, 120), "borderline": (80, 140)},
        "BP_DIA": {"safe": (60, 80),  "borderline": (50, 95)},
        "TEMP":   {"safe": (36.1, 37.5), "borderline": (35.8, 38.0)},
    }

    def __init__(self, predictor, data_manager):
        self.predictor = predictor
        self.data_manager = data_manager
        self.twins: Dict[str, Dict[str, Any]] = {}

    # -------------------- Public API --------------------

    def update_twin(self, patient_id: str, vitals: List[dict] | List[Any], predictions: Optional[List[dict]] = None):
        """
        Store vitals + predictions in the twin. If vitals are missing/partial,
        fetch latest values from DB for canonical channels.
        """
        latest = self._extract_latest_from_vitals(vitals)

        # Fill missing channels from DB
        needed = [k for k in ("ECG", "SpO2", "BP_SYS", "BP_DIA", "TEMP") if k not in latest]
        if needed:
            db_latest = self.data_manager.get_latest_values(patient_id, needed)
            for k, v in db_latest.items():
                if v is not None:
                    latest[k] = v

        preds = predictions or []
        risk_score = self._compute_risk(latest, preds)

        self.twins[patient_id] = {
            "patient_id": patient_id,
            "latest": latest,                # dict of canonical vitals
            "predictions": preds,            # as provided by your pipeline
            "risk_score": round(risk_score, 3),
            "updated_ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_twin(self, patient_id: str) -> Dict[str, Any]:
        return self.twins.get(patient_id, {})

    def get_all_twins_summary(self) -> Dict[str, Any]:
        total = len(self.twins)
        high_risk = [pid for pid, tw in self.twins.items() if (tw.get("risk_score") or 0) >= 0.8]
        return {"total_patients": total, "high_risk_patients": high_risk}

    # -------------------- Helpers --------------------

    def _extract_latest_from_vitals(self, vitals: List[dict] | List[Any]) -> Dict[str, float]:
        """
        Accepts vitals as list of dicts or objects:
          - keys/attrs: sensor_type/sensor/type + value
          - BP string like "120/80" supported (mapped to BP_SYS/BP_DIA)
        Returns canonical map: {ECG, SpO2, BP_SYS, BP_DIA, TEMP}
        """
        out: Dict[str, float] = {}
        for v in vitals or []:
            sensor = getattr(v, "sensor_type", getattr(v, "sensor", None))
            if sensor is None and isinstance(v, dict):
                sensor = v.get("sensor_type") or v.get("sensor") or v.get("type")
            value = getattr(v, "value", None)
            if value is None and isinstance(v, dict):
                value = v.get("value")
            if not sensor:
                continue

            canon = self._canon(sensor)

            # BP as "120/80"
            if canon == "BP" and isinstance(value, str) and "/" in value:
                try:
                    sys_v, dia_v = value.split("/")
                    out["BP_SYS"] = float(sys_v.strip())
                    out["BP_DIA"] = float(dia_v.strip())
                except Exception:
                    pass
                continue

            # Numeric channels
            if canon in ("ECG", "SpO2", "BP_SYS", "BP_DIA", "TEMP"):
                try:
                    out[canon] = float(value)
                except (TypeError, ValueError):
                    continue
        return out

    def _compute_risk(self, latest: Dict[str, float], predictions: List[dict]) -> float:
        """
        Simple interpretable heuristic:
          • SpO2 < 92 → +0.5
          • BP_SYS > 150 → +0.3
          • ECG < 50 or > 120 → +0.2
          • TEMP > 38.0 → +0.1
          • Forecasted systolic > 150 in any prediction → +0.2
        Clipped to [0,1].
        """
        risk = 0.0

        spo2 = latest.get("SpO2")
        if spo2 is not None and spo2 < 92:
            risk += 0.5

        sbp = latest.get("BP_SYS")
        if sbp is not None and sbp > 150:
            risk += 0.3

        ecg = latest.get("ECG")
        if ecg is not None and (ecg < 50 or ecg > 120):
            risk += 0.2

        temp = latest.get("TEMP")
        if temp is not None and temp > 38.0:
            risk += 0.1

        # Predictions contribution
        for p in predictions or []:
            # Accept either {"systolic_bp": 152.3} or {"systolic_bp": {"p50": 152.3}}
            val = None
            if isinstance(p, dict):
                sbp_pred = p.get("systolic_bp")
                if isinstance(sbp_pred, (int, float)):
                    val = float(sbp_pred)
                elif isinstance(sbp_pred, dict):
                    val = sbp_pred.get("p50") or sbp_pred.get("point")
            if val and val > 150:
                risk += 0.2
                break  # one hit is enough for now

        return max(0.0, min(1.0, risk))

    @classmethod
    def _canon(cls, name: str) -> str:
        key = (name or "").strip().lower()
        return cls.CANON_MAP.get(key, key.upper())

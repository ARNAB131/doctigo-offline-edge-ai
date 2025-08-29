#!/usr/bin/env python3
"""
Merged Doctigo Edge AI + Medical Forecasting Studio
Updated: 2025-08-22

Author: ECE Student
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
import asyncio

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --------------------------
# DOCTIGO EDGE CORE IMPORTS
# --------------------------
from edge_core import ProductionConfig
from edge_core.storage import init_sqlite_if_needed  # NEW
from edge_core import (
    DataManager,
    ProductionVitalsPredictor,
    DigitalTwinManager,
    AlertManager,
)
from edge_core import (
    SimulatedECGSensor,
    SimulatedPulseOximeter,
    SimulatedBloodPressureMonitor,
)
from utils.auth import login   # ENTER the revised version!
from utils.pdf_report import generate_pdf  # noqa: F401 (optional)
from utils.cloud_sync import simulate_sync  # noqa: F401 (optional)

# --------------------------
# LOGGING
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================
#               FORECASTING STUDIO CLASSES
# =============================================================

class MedicalDataGenerator:
    """Generate realistic BP + Temperature data with circadian patterns."""
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        self.random_state = 42
        np.random.seed(self.random_state)

    def generate_realistic_data(self) -> pd.DataFrame:
        start_time = datetime.now() - timedelta(hours=17)
        timestamps = [start_time + timedelta(minutes=10 * i) for i in range(self.n_samples)]
        time_hours = np.array([(ts.hour + ts.minute / 60) for ts in timestamps])

        systolic_base = 120 + 15 * np.sin(2 * np.pi * time_hours / 24 - np.pi / 4)
        systolic_noise = np.random.normal(0, 8, self.n_samples)
        systolic_bp = np.clip(systolic_base + systolic_noise, 85, 180)

        diastolic_bp = systolic_bp * (0.65 + 0.1 * np.random.normal(0, 0.1, self.n_samples))
        diastolic_bp = np.clip(diastolic_bp, 50, 110)

        temp_base = 36.7 + 0.4 * np.sin(2 * np.pi * time_hours / 24 - np.pi / 3)
        temp_noise = np.random.normal(0, 0.2, self.n_samples)
        body_temp = np.clip(temp_base + temp_noise, 35.5, 38.5)

        # mild positive correlation
        corr = 0.3
        systolic_bp += corr * (body_temp - 36.7) * 2
        diastolic_bp += corr * (body_temp - 36.7) * 1.5

        df = pd.DataFrame({
            "timestamp": timestamps,
            "systolic_bp": np.round(systolic_bp, 1),
            "diastolic_bp": np.round(diastolic_bp, 1),
            "body_temperature": np.round(body_temp, 1),
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

class MedicalDataForecaster:
    """Multi-target forecaster for systolic, diastolic, and temperature."""

    def __init__(self):
        self.models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=120, random_state=42),
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns: list[str] = []
        self.trained_models: dict[str, dict[str, object]] = {}

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["dow"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        for col in ["systolic_bp", "diastolic_bp", "body_temperature"]:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)
            df[f"{col}_lag3"] = df[col].shift(3)
            df[f"{col}_roll_mean"] = df[col].rolling(window=5, min_periods=1).mean()
            df[f"{col}_roll_std"] = df[col].rolling(window=5, min_periods=1).std()
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def _prepare(self, data: pd.DataFrame, targets: list[str]):
        feats = self._create_features(data)
        exclude = ["timestamp"] + targets
        self.feature_columns = [c for c in feats.columns if c not in exclude]
        X = feats[self.feature_columns].values
        y = feats[targets].values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        return X, y

    def train(self, data: pd.DataFrame):
        targets = ["systolic_bp", "diastolic_bp", "body_temperature"]
        X, y = self._prepare(data, targets)
        Xs = self.scaler.fit_transform(X)
        self.trained_models = {}
        for name, model in self.models.items():
            self.trained_models[name] = {}
            for i, tgt in enumerate(targets):
                mdl = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
                mdl.fit(Xs, y[:, i])
                self.trained_models[name][tgt] = mdl
        self.is_fitted = True

    def forecast(self, data: pd.DataFrame, hours_ahead: list[int] = [3, 6, 9]) -> dict:
        if not self.is_fitted:
            raise ValueError("Models must be trained before forecasting")
        targets = ["systolic_bp", "diastolic_bp", "body_temperature"]
        last_ts = pd.to_datetime(data["timestamp"].iloc[-1])
        out = {}
        for h in hours_ahead:
            future_ts = last_ts + timedelta(hours=h)
            future_row = data.iloc[-1:].copy()
            future_row["timestamp"] = future_ts
            ext = pd.concat([data, future_row], ignore_index=True)
            Xf, _ = self._prepare(ext, targets)
            Xf = self.scaler.transform(Xf)
            Xp = Xf[-1:, :]
            pred = {}
            for tgt in targets:
                preds = []
                for mname in self.trained_models:
                    preds.append(self.trained_models[mname][tgt].predict(Xp)[0])
                pred[tgt] = float(np.mean(preds))
            out[f"{h}h"] = pred
        return out

class MedicalDataVisualizer:
    def time_series(self, data: pd.DataFrame) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=["Blood Pressure Over Time", "Temperature Over Time", "BP vs Temperature"],
            vertical_spacing=0.08,
        )
        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data["systolic_bp"], name="Systolic BP", line=dict(width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data["diastolic_bp"], name="Diastolic BP", line=dict(width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data["body_temperature"], name="Body Temperature", line=dict(width=2)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["body_temperature"],
                y=data["systolic_bp"],
                mode="markers",
                name="Systolic vs Temp",
                marker=dict(size=6, opacity=0.6),
            ),
            row=3,
            col=1,
        )
        fig.update_layout(height=800, showlegend=True, title_text="Medical Data Analysis")
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Body Temperature (°C)", row=3, col=1)
        fig.update_yaxes(title_text="Blood Pressure (mmHg)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Systolic BP (mmHg)", row=3, col=1)
        return fig

    def distributions(self, data: pd.DataFrame) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Systolic BP Distribution",
                "Diastolic BP Distribution",
                "Temperature Distribution",
                "Combined Statistics",
            ],
        )
        fig.add_trace(
            go.Histogram(x=data["systolic_bp"], nbinsx=20, name="Systolic BP", opacity=0.7, histnorm="probability density"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=data["diastolic_bp"], nbinsx=20, name="Diastolic BP", opacity=0.7, histnorm="probability density"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(x=data["body_temperature"], nbinsx=20, name="Temperature", opacity=0.7, histnorm="probability density"),
            row=2,
            col=1,
        )
        fig.add_trace(go.Box(y=data["systolic_bp"], name="Systolic"), row=2, col=2)
        fig.add_trace(go.Box(y=data["diastolic_bp"], name="Diastolic"), row=2, col=2)
        fig.add_trace(go.Box(y=data["body_temperature"], name="Temperature"), row=2, col=2)
        fig.update_layout(height=600, showlegend=True, title_text="Data Distributions")
        return fig

    def corr_heatmap(self, data: pd.DataFrame) -> go.Figure:
        corr = data[["systolic_bp", "diastolic_bp", "body_temperature"]].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
            )
        )
        fig.update_layout(title="Correlation Matrix of Medical Parameters", height=400)
        return fig

# =============================================================
#                     STREAMLIT APP
# =============================================================

def main():
    st.set_page_config(page_title="🧠 Doctigo Edge AI + Forecasting Studio", page_icon="🩺", layout="wide")
    st.title("🩺 Real-Time Edge AI Patient Monitoring + 🔮 Forecasting Studio")

    # Role-Based AUTH
    if not login():
        st.stop()

    user_role = st.session_state.get("user_role", None)
    if user_role:
        st.sidebar.success(f"Logged in as: {user_role}")

    # Needed for all tabs below (config + DB init)
    patient_id = "patient_001"
    device_id = "edge_001"

    # Load OFFLINE settings and ensure local DB exists
    config = ProductionConfig.from_settings()     # CHANGED
    init_sqlite_if_needed(config.sqlite_path)     # NEW

    data_manager = DataManager(config)
    predictor = ProductionVitalsPredictor(config)
    twin_manager = DigitalTwinManager(predictor, data_manager)
    alert_manager = AlertManager(config, data_manager)
    ecg_sensor = SimulatedECGSensor(patient_id, device_id)
    spo2_sensor = SimulatedPulseOximeter(patient_id, device_id)
    bp_sensor = SimulatedBloodPressureMonitor(patient_id, device_id)

    # Guard any cloud sync when offline
    if not getattr(config, "offline", True):
        try:
            simulate_sync()
        except Exception:
            pass

    # Helpful sidebar status
    st.sidebar.caption(f"Mode: {'OFFLINE' if config.offline else 'ONLINE'} · DB: {config.sqlite_path}")

    # Set up role-based tabs
    tab_names: list[str] = []
    if user_role == "Admin":
        tab_names = [
            "👑 Admin Dashboard", "📡 Live Monitor", "📤 CSV Upload + Batch Predict",
            "🔮 Forecasting Studio", "📊 Statistical Analysis",
            "📈 Visualizations", "📋 Raw Data"
        ]
    elif user_role == "Doctor":
        tab_names = [
            "📡 Live Monitor", "📤 CSV Upload + Batch Predict",
            "🔮 Forecasting Studio", "📊 Statistical Analysis",
            "📈 Visualizations", "📋 Raw Data"
        ]
    elif user_role == "Hospital":
        tab_names = [
            "📡 Live Monitor", "📊 Statistical Analysis", "📈 Visualizations"
        ]
    else:
        tab_names = [
            "📡 Live Monitor", "📤 CSV Upload + Batch Predict",
            "🔮 Forecasting Studio", "📊 Statistical Analysis",
            "📈 Visualizations", "📋 Raw Data"
        ]

    tabs = st.tabs(tab_names)

    # =============== TAB LOGIC ===============

    # Admin-only dashboard
    if "👑 Admin Dashboard" in tab_names:
        with tabs[tab_names.index("👑 Admin Dashboard")]:
            st.subheader("👑 Admin Dashboard")
            st.info("Admin-only settings and patient/user controls go here.")

    # Live Monitor Tab
    if "📡 Live Monitor" in tab_names:
        with tabs[tab_names.index("📡 Live Monitor")]:
            st.subheader("📡 Live Vitals Monitoring")
            auto_refresh = st.checkbox("Enable Auto Mode", value=True)
            st.slider("Refresh Interval (seconds)", 1, 10, 3)  # currently not used as a timer

            graph_ph = st.empty()
            desc_ph = st.empty()
            gauge_ph = st.empty()

            def one_cycle_read_and_plot():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                vitals = []
                use_ecg = True
                use_spo2 = True
                use_bp = True

                if use_ecg:
                    v = loop.run_until_complete(ecg_sensor.read_data())
                    vitals.append(v)
                if use_spo2:
                    v = loop.run_until_complete(spo2_sensor.read_data())
                    vitals.append(v)
                if use_bp:
                    bp_readings = loop.run_until_complete(bp_sensor.read_data())
                    if isinstance(bp_readings, list):
                        vitals.extend(bp_readings)
                    else:
                        vitals.append(bp_readings)

                for v in vitals:
                    data_manager.store_vital_sign(v)

                df = pd.DataFrame(data_manager.get_patient_vitals_history(patient_id, limit=50))
                if df.empty:
                    desc_ph.info("No vitals data available yet.")
                    return
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                fig = go.Figure()
                alert_messages: list[str] = []
                latest_values_for_plot: dict[str, float] = {}

                sensor_colors = {"ECG": "red", "SpO2": "blue", "BP_SYS": "green", "BP_DIA": "orange"}
                ranges = {
                    "ECG": {"safe": (60, 100), "borderline": (50, 110)},
                    "SpO2": {"safe": (95, 100), "borderline": (90, 94)},
                    "BP_SYS": {"safe": (90, 130), "borderline": (80, 140)},
                    "BP_DIA": {"safe": (60, 90), "borderline": (50, 95)},
                }

                for sensor in ["ECG", "SpO2", "BP_SYS", "BP_DIA"]:
                    s = df[df["sensor"] == sensor]
                    if s.empty:
                        continue
                    y = pd.to_numeric(s["value"], errors="coerce")
                    latest_values_for_plot[sensor] = y.iloc[-1]
                    fig.add_trace(
                        go.Scatter(x=s["timestamp"], y=y, mode="lines+markers", name=sensor, line=dict(color=sensor_colors[sensor]))
                    )
                    safe_low, safe_high = ranges[sensor]["safe"]
                    border_low, border_high = ranges[sensor]["borderline"]
                    fig.add_hrect(y0=safe_low, y1=safe_high, fillcolor="green", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=border_low, y1=safe_low, fillcolor="yellow", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=safe_high, y1=border_high, fillcolor="yellow", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=border_high, y1=max(y) + 10, fillcolor="red", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=min(y) - 10, y1=border_low, fillcolor="red", opacity=0.1, line_width=0)

                    val = latest_values_for_plot[sensor]
                    if val < border_low or val > border_high:
                        alert_messages.append(f"🚨 {sensor}: {val} (Critical)")
                    elif val < safe_low or val > safe_high:
                        alert_messages.append(f"⚠️ {sensor}: {val} (Borderline)")

                fig.update_layout(
                    title="📈 ICU Live Multi-Sensor Monitor",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    xaxis=dict(tickformat="%H:%M:%S"),
                    legend=dict(orientation="h", y=-0.2),
                )
                graph_ph.plotly_chart(fig, use_container_width=True)

                if alert_messages:
                    desc_ph.error("\n".join(alert_messages))
                    # Optional local toasts (offline-friendly)
                    if getattr(config, "alerts_toast", True):
                        for m in alert_messages:
                            st.toast(m)
                else:
                    desc_ph.success("✅ All vitals in safe range.")

                # DB-backed latest values for gauges (robust, fast)
                latest_values = data_manager.get_latest_values(patient_id, ["ECG", "SpO2", "BP_SYS", "BP_DIA"])

                gauge_cols = gauge_ph.columns(4)
                for i, sensor in enumerate(["ECG", "SpO2", "BP_SYS", "BP_DIA"]):
                    val = latest_values.get(sensor)
                    if val is not None:
                        color = sensor_colors[sensor]
                        gauge_cols[i].markdown(
                            f"""
                            <div style='text-align:center;padding:10px;background:#000;border-radius:10px;'>
                                <h4 style='color:white'>{sensor}</h4>
                                <h2 style='color:{color}'>{val:.1f}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if st.button("📈 Read Selected Sensors Once"):
                one_cycle_read_and_plot()
            if auto_refresh:
                st.info("Auto-refresh enabled. Click the button above to refresh vitals manually.")

            # Update digital twin & alerts (non-blocking)
            hist = data_manager.get_patient_vitals_history(patient_id, limit=30)
            pred_trend = predictor.predict_trend(patient_id, hist)
            tw_preds = [pred_trend] if pred_trend else []
            twin_manager.update_twin(patient_id, [], tw_preds)
            alert_manager.generate_alert(patient_id, twin_manager.get_twin(patient_id), tw_preds)

    # CSV Upload/Batch Predictions Tab
    if "📤 CSV Upload + Batch Predict" in tab_names:
        with tabs[tab_names.index("📤 CSV Upload + Batch Predict")]:
            st.subheader("📤 Upload Vitals CSV for Batch Predictions")
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.success("File uploaded!")
                    st.dataframe(df_up, use_container_width=True)
                    # (Optional) add batch import -> DataManager.bulk_store_vitals(...)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                st.info("Please upload a CSV file to display and predict.")

    # Forecasting Studio Tab
    if "🔮 Forecasting Studio" in tab_names:
        with tabs[tab_names.index("🔮 Forecasting Studio")]:
            st.subheader("🔮 Forecasting Studio (BP + Temperature)")
            st.caption("Generate realistic time series or bring your own CSV; train multi-model ensemble and forecast 3/6/9 hours ahead.")

            colA, colB = st.columns([1, 2])
            with colA:
                source = st.radio("Data source", ["Generate synthetic", "Upload CSV"], index=0)
                n_samples = st.slider("Synthetic samples", 50, 200, 100)
                go_button = st.button("🚀 Prepare Data")
            with colB:
                uploader = st.file_uploader("(Optional) Upload CSV with columns: timestamp, systolic_bp, diastolic_bp, body_temperature", type=["csv"], key="forecast_csv")

            if go_button or uploader:
                if source == "Generate synthetic" or (uploader is None):
                    data = MedicalDataGenerator(n_samples).generate_realistic_data()
                else:
                    try:
                        data = pd.read_csv(uploader)
                        if "timestamp" in data.columns:
                            data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        st.error(f"Data upload error: {e}")
                        data = pd.DataFrame()
                st.session_state["forecast_data"] = data
                if not data.empty:
                    st.success(f"Data ready with {len(data)} rows.")
                    st.dataframe(data, use_container_width=True)
                else:
                    st.info("No data available.")

            data = st.session_state.get("forecast_data", pd.DataFrame())
            if not data.empty:
                if st.button("🧠 Train models & Generate 3/6/9h Forecasts", type="primary"):
                    try:
                        forecaster = MedicalDataForecaster()
                        forecaster.train(data)
                        fc = forecaster.forecast(data)
                        st.session_state["forecasts"] = fc
                        st.success("Forecasts generated.")

                        cols = st.columns(3)
                        for i, (h, p) in enumerate(fc.items()):
                            with cols[i]:
                                st.markdown(f"**{h} Forecast**")
                                st.metric("Systolic BP", f"{p['systolic_bp']:.1f} mmHg")
                                st.metric("Diastolic BP", f"{p['diastolic_bp']:.1f} mmHg")
                                st.metric("Temperature", f"{p['body_temperature']:.1f}°C")
                    except Exception as e:
                        st.error(f"Forecasting error: {e}")

    # Statistical Analysis Tab
    if "📊 Statistical Analysis" in tab_names:
        with tabs[tab_names.index("📊 Statistical Analysis")]:
            st.subheader("📊 Statistical Analysis (on Forecasting Studio data)")
            data = st.session_state.get("forecast_data", pd.DataFrame())
            if data.empty:
                st.info("Use the 🔮 Forecasting Studio tab to prepare data first.")
            else:
                viz = MedicalDataVisualizer()
                st.plotly_chart(viz.distributions(data), use_container_width=True)
                st.plotly_chart(viz.corr_heatmap(data), use_container_width=True)
                st.subheader("Statistical Summary")
                st.dataframe(data.describe(), use_container_width=True)

    # Visualizations Tab
    if "📈 Visualizations" in tab_names:
        with tabs[tab_names.index("📈 Visualizations")]:
            st.subheader("📈 Time Series & Relationships (on Forecasting Studio data)")
            data = st.session_state.get("forecast_data", pd.DataFrame())
            if data.empty:
                st.info("Use the 🔮 Forecasting Studio tab to prepare data first.")
            else:
                viz = MedicalDataVisualizer()
                st.plotly_chart(viz.time_series(data), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    bp_cats = pd.cut(
                        data["systolic_bp"],
                        bins=[0, 120, 140, 180, float("inf")],
                        labels=["Normal", "Elevated", "High", "Very High"],
                    )
                    counts = bp_cats.value_counts()
                    fig_pie = px.pie(values=counts.values, names=counts.index, title="Blood Pressure Categories Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    temp_ranges = pd.cut(
                        data["body_temperature"],
                        bins=[35, 36.5, 37.2, 38, float("inf")],
                        labels=["Low", "Normal", "Elevated", "Fever"],
                    )
                    tcounts = temp_ranges.value_counts()
                    fig_bar = px.bar(x=tcounts.index, y=tcounts.values, title="Temperature Range Distribution")
                    st.plotly_chart(fig_bar, use_container_width=True)

    # Raw Data Tab
    if "📋 Raw Data" in tab_names:
        with tabs[tab_names.index("📋 Raw Data")]:
            st.subheader("📋 Raw Data (from Forecasting Studio)")
            data = st.session_state.get("forecast_data", pd.DataFrame())
            if data.empty:
                st.info("Use the 🔮 Forecasting Studio tab to prepare data first.")
            else:
                st.dataframe(data, use_container_width=True)
                st.download_button(
                    label="Download data as CSV",
                    data=data.to_csv(index=False),
                    file_name=f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    # SIDEBAR SUMMARY
    st.sidebar.title("📋 Summary")
    summary = twin_manager.get_all_twins_summary()
    alerts = alert_manager.get_alert_statistics()
    st.sidebar.metric("Total Patients", summary.get("total_patients", 0))
    st.sidebar.metric("Active Alerts", alerts.get("active_alerts", 0))
    st.sidebar.metric("High Risk", len(summary.get("high_risk_patients", [])))

if __name__ == "__main__":
    main()

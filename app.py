import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download
from sgp4.api import Satrec, jday

# Page Configuration
st.set_page_config(
    page_title="Space Debris AI Predictor",
    page_icon="🛰️",
    layout="wide"
)

# -------------------------------------------------------------
# 1. MODEL 1 PYTORCH RESNET ARCHITECTURE
# -------------------------------------------------------------
class DeepOrbitResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 128)
        self.drop = nn.Dropout(0.1)
        self.block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        self.out_proj = nn.Linear(128, 3)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.in_proj(x))
        h = self.act(h + self.block1(h))
        h = self.act(h + self.block2(h))
        return self.out_proj(self.drop(h))

# -------------------------------------------------------------
# 2. CACHED MODEL LOADING FROM HUGGING FACE
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    m1_weights = hf_hub_download(repo_id="RootCode26/sgp4-propagation-corrector", filename="model1_propagation_corrector.pt")
    scaler_x_path = hf_hub_download(repo_id="RootCode26/sgp4-propagation-corrector", filename="scaler_model1.pkl")
    scaler_y_path = hf_hub_download(repo_id="RootCode26/sgp4-propagation-corrector", filename="scaler_target_model1.pkl")
    
    m1 = DeepOrbitResNet()
    m1.load_state_dict(torch.load(m1_weights, map_location="cpu"))
    m1.eval()
    
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    m2_path = hf_hub_download(repo_id="RootCode26/orbit-conjunction-classifier", filename="model2_conjunction_classifier.pkl")
    m2 = joblib.load(m2_path)
    return m1, scaler_x, scaler_y, m2

model1, scaler_x, scaler_y, model2 = load_models()

PRESET_TLES = {
    "ISS (ZARYA)": (
        "1 25544U 98067A   24235.51234567  .00012345  00000-0  22345-3 0  9993",
        "2 25544  51.6432 120.1234 0004123  45.1234 315.1234 15.49812345441234"
    ),
    "COSMOS DEBRIS": (
        "1 48274U 21035A   24235.54321098  .00023456  00000-0  34567-3 0  9997",
        "2 48274  51.6490 120.1250 0004150  45.1300 315.1200 15.49815000182345"
    ),
    "STARLINK-1007": (
        "1 44713U 19074A   24235.45678901  .00001234  00000-0  12345-4 0  9991",
        "2 44713  53.0540 312.4567 0001234 110.1234 250.1234 15.06412345261234"
    )
}

# -------------------------------------------------------------
# 3. STREAMLIT UI
# -------------------------------------------------------------
st.title("🛰️ Space Debris Tracking & Collision Risk Engine")
st.caption("Real-time inference using **Model 1** (SGP4 Residual Corrector) & **Model 2** (Conjunction Classifier).")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Conjunction Setup")
    sat_a_choice = st.selectbox("Primary Spacecraft", list(PRESET_TLES.keys()), index=0)
    sat_b_choice = st.selectbox("Approaching Debris", list(PRESET_TLES.keys()), index=1)
    
    miss_dist = st.slider("Miss Distance (meters)", min_value=5.0, max_value=5000.0, value=250.0, step=5.0)
    rel_vel = st.slider("Relative Velocity (km/s)", min_value=1.0, max_value=16.0, value=11.5, step=0.1)
    radius = st.slider("Combined Hard-Body Radius (m)", min_value=1.0, max_value=25.0, value=15.0, step=0.5)
    
    run_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

with col2:
    if run_btn:
        # 1. SGP4 Base Propagation
        l1_a, l2_a = PRESET_TLES[sat_a_choice]
        l1_b, l2_b = PRESET_TLES[sat_b_choice]
        
        now = datetime.now(timezone.utc)
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
        sat_a = Satrec.twoline2rv(l1_a, l2_a)
        sat_b = Satrec.twoline2rv(l1_b, l2_b)
        
        _, pos_a, vel_a = sat_a.sgp4(jd, fr)
        _, pos_b, vel_b = sat_b.sgp4(jd, fr)
        pos_a, vel_a = np.array(pos_a), np.array(vel_a)
        pos_b, vel_b = np.array(pos_b), np.array(vel_b)
        
        alt_a = np.linalg.norm(pos_a) - 6378.137
        epoch_f = getattr(sat_a, 'jdsatepochF', getattr(sat_a, 'jdsatepochf', 0.0))
        time_since_epoch = ((jd + fr) - (sat_a.jdsatepoch + epoch_f)) * 24.0
        inc_a = np.degrees(sat_a.inclo)
        
        # 2. Model 1 Prediction (Unit Fixed: kilometers)
        feat_m1 = np.array([[pos_a[0], pos_a[1], pos_a[2], vel_a[0], vel_a[1], vel_a[2], alt_a, sat_a.bstar, time_since_epoch, inc_a]])
        scaled_in = scaler_x.transform(feat_m1)
        
        with torch.no_grad():
            in_tensor = torch.tensor(scaled_in, dtype=torch.float32)
            pred_scaled = model1(in_tensor).numpy()
            dx_km, dy_km, dz_km = scaler_y.inverse_transform(pred_scaled)[0]
        
        # Consistent Units (km + km)
        refined_pos_a = pos_a + np.array([dx_km, dy_km, dz_km])
        
        # 3. Model 2 Conjunction Assessment
        radial_sep = miss_dist * 0.15
        in_track_sep = miss_dist * 0.65
        cross_track_sep = np.sqrt(max(0, miss_dist**2 - radial_sep**2 - in_track_sep**2))
        sigma_r, sigma_t, sigma_n = 25.0, 100.0, 50.0
        
        feat_m2 = np.array([[miss_dist, radial_sep, in_track_sep, cross_track_sep, rel_vel, radius, alt_a, sigma_r, sigma_t, sigma_n]])
        probs = model2.predict_proba(feat_m2)[0]
        label_map = {0: "LOW", 1: "MEDIUM", 2: "CRITICAL"}
        pred_label = label_map[int(np.argmax(probs))]
        
        # Physical 2D Collision Probability Formulation (in %)
        combined_sigma_sq = (sigma_r**2 + sigma_t**2 + sigma_n**2) / 3.0
        u_sq = (miss_dist**2) / combined_sigma_sq
        p_c_raw = np.exp(-0.5 * u_sq) * (1.0 - np.exp(-0.5 * (radius**2 / combined_sigma_sq)))
        collision_probability_pct = min(100.0, p_c_raw * 100.0)

        # Dashboard Summary Cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            badge_color = "🟢" if pred_label == "LOW" else ("🟠" if pred_label == "MEDIUM" else "🔴")
            st.metric(label="Risk Assessment", value=f"{badge_color} {pred_label}")
        with metric_col2:
            st.metric(label="Collision Probability", value=f"{collision_probability_pct:.4f}%")
        with metric_col3:
            total_drift_m = np.linalg.norm([dx_km, dy_km, dz_km]) * 1000.0
            st.metric(label="Total Predicted Drift", value=f"{total_drift_m:.2f} m")
            
        # 4. 3D Visualization
        fig = go.Figure()
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        x_earth = 6378.137 * np.cos(u) * np.sin(v)
        y_earth = 6378.137 * np.sin(u) * np.sin(v)
        z_earth = 6378.137 * np.cos(v)
        fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.35, name="Earth"))
        
        fig.add_trace(go.Scatter3d(x=[pos_a[0]], y=[pos_a[1]], z=[pos_a[2]], mode='markers+text', name='Raw SGP4', marker=dict(size=6, color='yellow'), text=['Raw SGP4']))
        fig.add_trace(go.Scatter3d(x=[refined_pos_a[0]], y=[refined_pos_a[1]], z=[refined_pos_a[2]], mode='markers+text', name='Model 1 Refined', marker=dict(size=8, color='cyan'), text=['Refined Orbit']))
        fig.add_trace(go.Scatter3d(x=[pos_b[0]], y=[pos_b[1]], z=[pos_b[2]], mode='markers+text', name=sat_b_choice, marker=dict(size=8, color='red'), text=['Debris Target']))
        
        fig.update_layout(scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'), height=520, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Diagnostic JSON Out
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Model 1 Drift Output")
            st.json({
                "Raw Position (km)": pos_a.round(3).tolist(),
                "Predicted Drift [dx, dy, dz] (meters)": [round(dx_km * 1000.0, 2), round(dy_km * 1000.0, 2), round(dz_km * 1000.0, 2)],
                "Refined Position (km)": refined_pos_a.round(3).tolist()
            })
        with res_col2:
            st.subheader("Model 2 Risk Assessment")
            st.json({
                "Risk Classification": pred_label,
                "Estimated Collision Probability": f"{collision_probability_pct:.6f}%",
                "Class Confidence Breakdown": {
                    "LOW": f"{probs[0]*100:.2f}%",
                    "MEDIUM": f"{probs[1]*100:.2f}%",
                    "CRITICAL": f"{probs[2]*100:.2f}%"
                }
            })
    else:
        st.info("Select encounter parameters and click **Run AI Analysis**.")

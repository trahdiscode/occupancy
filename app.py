# ============================================================
# Energy-Signature Based Occupancy Sensing — Software Demo
# Clean, minimal, professional dashboard UI.
#
# Uses simulated power readings (standing in for the ESP32 +
# sensor hardware, which isn't built yet) so the full software
# pipeline can be shown as progress.
#
# Once hardware is ready, only simulate_reading() needs to be
# replaced with a real Firebase/API fetch — everything else
# (feature extraction, classification, dashboard) stays the same.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ROOM_IDS = ["Room 1", "Room 2"]

HISTORY_LENGTH = 60          # readings kept per room, for the chart
WINDOW_SIZE = 10             # rolling window for feature extraction

BASELINE_POWER_THRESHOLD = 15.0   # Watts above idle baseline
VARIANCE_THRESHOLD = 5.0          # fluctuation level

st.set_page_config(
    page_title="Occupancy Sensing Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# STYLE: minimal, editorial, gradient-accented
# ------------------------------------------------------------
st.markdown("""
<style>

    html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

    .main {
        background-color: #0A0B0F;
    }
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 4rem;
        max-width: 1080px;
    }

    /* Hero header */
    .hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6C7CFF;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #F5F6FA;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #80869C;
        font-weight: 400;
        margin-bottom: 3rem;
        max-width: 620px;
        line-height: 1.5;
    }

    /* Room card */
    .room-card {
        background: linear-gradient(180deg, #12141C 0%, #0F1117 100%);
        border: 1px solid #1E2130;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        transition: border-color 0.2s ease;
    }
    .room-card:hover {
        border-color: #2A2E42;
    }
    .room-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F5F6FA;
        letter-spacing: -0.01em;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .status-occupied {
        background: linear-gradient(135deg, rgba(108,124,255,0.18), rgba(63,185,80,0.18));
        color: #6EE787;
        border: 1px solid rgba(110,231,135,0.35);
    }
    .status-unoccupied {
        background-color: rgba(128, 134, 156, 0.1);
        color: #80869C;
        border: 1px solid rgba(128,134,156,0.25);
    }

    /* Metrics */
    .metric-label {
        font-size: 0.68rem;
        color: #565C72;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.4rem;
        color: #F5F6FA;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .metric-block {
        margin-bottom: 1.3rem;
    }

    /* Divider line under header */
    .header-divider {
        border: none;
        border-top: 1px solid #1E2130;
        margin: 0 0 2.5rem 0;
    }

    /* Footer */
    .footer-note {
        font-size: 0.75rem;
        color: #4A4F63;
        margin-top: 2rem;
        text-align: center;
        letter-spacing: 0.02em;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">Innovative Design Project</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Room Occupancy Sensing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Non-intrusive occupancy detection via real-time energy-signature analysis. '
    'Currently running on simulated sensor data while hardware integration is in progress.</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="header-divider">', unsafe_allow_html=True)


# ------------------------------------------------------------
# STEP 1: Initialize session state to hold reading history
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {}

for room_id in ROOM_IDS:
    if room_id not in st.session_state.history:
        st.session_state.history[room_id] = []


# ------------------------------------------------------------
# STEP 2: Simulate a power reading for a room
# ------------------------------------------------------------
def simulate_reading(room_id):
    """
    Generates a fake power reading in Watts, mimicking what a real
    energy sensor would report.

    Logic: randomly decide if this room is 'occupied' right now,
    then generate power values consistent with that state:
      - Unoccupied: low, flat baseline power (~5-10W, just standby loads)
      - Occupied: higher average power with more fluctuation
        (lights, fan, laptop charging, etc. switching on/off)

    This is only a placeholder until the real sensor feeds actual data.
    """
    simulated_occupied = random.random() < 0.5

    if simulated_occupied:
        base = random.uniform(20, 40)
        noise = random.uniform(-8, 8)
    else:
        base = random.uniform(4, 9)
        noise = random.uniform(-1, 1)

    return max(0.0, base + noise)


# ------------------------------------------------------------
# STEP 3: Compute rolling-window features (the "energy signature")
# ------------------------------------------------------------
def compute_features(readings, window_size=WINDOW_SIZE):
    """
    Takes the most recent `window_size` readings and computes:
      - mean power
      - variance (fluctuation)
      - number of significant peaks (spikes above mean)
    These three numbers together form the room's current energy signature.
    """
    recent = readings[-window_size:] if len(readings) >= window_size else readings
    recent_array = np.array(recent)

    mean_power = recent_array.mean()
    variance = recent_array.var() if len(recent_array) > 1 else 0.0
    peak_count = int(np.sum(recent_array > (mean_power + variance)))

    return {
        "mean_power": mean_power,
        "variance": variance,
        "peak_count": peak_count
    }


# ------------------------------------------------------------
# STEP 4: Classify occupancy using simple threshold rules
# ------------------------------------------------------------
def classify_occupancy(features):
    """
    A room is 'Occupied' if its average power draw is meaningfully
    above idle baseline AND there's enough fluctuation to suggest
    active usage (not just a fridge or router humming at constant load).
    """
    is_high_power = features["mean_power"] > BASELINE_POWER_THRESHOLD
    is_fluctuating = features["variance"] > VARIANCE_THRESHOLD
    return "Occupied" if (is_high_power and is_fluctuating) else "Unoccupied"


# ------------------------------------------------------------
# Auto-refreshing fragment: updates readings + redraws dashboard
# every 2 seconds, without re-running the entire script/page.
# ------------------------------------------------------------
@st.fragment(run_every=2)
def render_dashboard():
    for room_id in ROOM_IDS:
        new_reading = simulate_reading(room_id)
        st.session_state.history[room_id].append(new_reading)

        if len(st.session_state.history[room_id]) > HISTORY_LENGTH:
            st.session_state.history[room_id] = st.session_state.history[room_id][-HISTORY_LENGTH:]

    for room_id in ROOM_IDS:
        readings = st.session_state.history[room_id]
        features = compute_features(readings)
        status = classify_occupancy(features)
        status_class = "status-occupied" if status == "Occupied" else "status-unoccupied"

        st.markdown('<div class="room-card">', unsafe_allow_html=True)

        header_col, status_col = st.columns([4, 1])
        with header_col:
            st.markdown(f'<div class="room-name">{room_id}</div>', unsafe_allow_html=True)
        with status_col:
            st.markdown(
                f'<div style="text-align:right;"><span class="status-pill {status_class}">{status}</span></div>',
                unsafe_allow_html=True
            )

        st.write("")

        metric_col, chart_col = st.columns([1, 3])
        with metric_col:
            st.markdown(f"""
                <div class="metric-block">
                    <div class="metric-label">Avg Power</div>
                    <div class="metric-value">{features['mean_power']:.1f} W</div>
                </div>
                <div class="metric-block">
                    <div class="metric-label">Variance</div>
                    <div class="metric-value">{features['variance']:.1f}</div>
                </div>
                <div class="metric-block">
                    <div class="metric-label">Peak Count</div>
                    <div class="metric-value">{features['peak_count']}</div>
                </div>
            """, unsafe_allow_html=True)
        with chart_col:
            chart_df = pd.DataFrame({"Power (W)": readings})
            st.line_chart(chart_df, height=190)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">Auto-refreshing every 2 seconds &nbsp;·&nbsp; Simulated data pending hardware integration</div>',
        unsafe_allow_html=True
    )

render_dashboard()

# ============================================================
# Energy-Signature Based Occupancy Sensing — Software Demo
# Minimalistic, professional dashboard UI.
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
# STYLE: minimalistic, professional CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .main {
        background-color: #0E1117;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Header */
    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #8A8F98;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Room card container */
    .room-card {
        background-color: #14161C;
        border: 1px solid #23262F;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .room-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 0.9rem;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .status-occupied {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3FB950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .status-unoccupied {
        background-color: rgba(139, 148, 158, 0.15);
        color: #8B949E;
        border: 1px solid rgba(139, 148, 158, 0.3);
    }

    /* Metric labels */
    .metric-label {
        font-size: 0.72rem;
        color: #6E7681;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-size: 1.25rem;
        color: #FAFAFA;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    /* Footer note */
    .footer-note {
        font-size: 0.78rem;
        color: #6E7681;
        margin-top: 1.5rem;
        text-align: center;
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
st.markdown('<div class="app-title">Room Occupancy Sensing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Energy-signature based detection — live monitoring (simulated data, hardware in progress)</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# STEP 1: Initialize session state to hold reading history
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {room_id: [] for room_id in ROOM_IDS}


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
# STEP 5: Update history with a new simulated reading per room
# ------------------------------------------------------------
for room_id in ROOM_IDS:
    new_reading = simulate_reading(room_id)
    st.session_state.history[room_id].append(new_reading)

    if len(st.session_state.history[room_id]) > HISTORY_LENGTH:
        st.session_state.history[room_id] = st.session_state.history[room_id][-HISTORY_LENGTH:]


# ------------------------------------------------------------
# STEP 6: Dashboard layout — one clean card per room
# ------------------------------------------------------------
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

    metric_col, chart_col = st.columns([1, 3])

    with metric_col:
        st.markdown(f"""
            <div class="metric-label">Avg Power</div>
            <div class="metric-value">{features['mean_power']:.1f} W</div>
            <br>
            <div class="metric-label">Variance</div>
            <div class="metric-value">{features['variance']:.1f}</div>
            <br>
            <div class="metric-label">Peak Count</div>
            <div class="metric-value">{features['peak_count']}</div>
        """, unsafe_allow_html=True)

    with chart_col:
        chart_df = pd.DataFrame({"Power (W)": readings})
        st.line_chart(chart_df, height=180)

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# STEP 7: Footer + auto-refresh
# ------------------------------------------------------------
st.markdown(
    '<div class="footer-note">Auto-refreshing every 2 seconds · Simulated data pending hardware integration</div>',
    unsafe_allow_html=True
)
time.sleep(2)
st.rerun()

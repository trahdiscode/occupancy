# ============================================================
# Energy-Signature Based Occupancy Sensing — Software Demo
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
ROOM_IDS = ["room_1", "room_2"]

# How many past readings to keep in memory per room, for the chart
HISTORY_LENGTH = 60

# Window size used to compute rolling features for classification
WINDOW_SIZE = 10

# Thresholds — placeholders, will be tuned once real sensor data exists
BASELINE_POWER_THRESHOLD = 15.0   # Watts above idle baseline
VARIANCE_THRESHOLD = 5.0          # fluctuation level

st.set_page_config(page_title="Energy-Signature Occupancy Sensing", layout="wide")
st.title("Energy-Signature Based Room Occupancy Sensing")
st.caption("Software demo — running on simulated sensor data (hardware in progress)")


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
        base = random.uniform(20, 40)       # higher baseline
        noise = random.uniform(-8, 8)        # more fluctuation
    else:
        base = random.uniform(4, 9)          # low idle baseline
        noise = random.uniform(-1, 1)        # very little fluctuation

    power = max(0.0, base + noise)
    return power


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

    if is_high_power and is_fluctuating:
        return "Occupied"
    else:
        return "Unoccupied"


# ------------------------------------------------------------
# STEP 5: Update history with a new simulated reading per room
# ------------------------------------------------------------
for room_id in ROOM_IDS:
    new_reading = simulate_reading(room_id)
    st.session_state.history[room_id].append(new_reading)

    if len(st.session_state.history[room_id]) > HISTORY_LENGTH:
        st.session_state.history[room_id] = st.session_state.history[room_id][-HISTORY_LENGTH:]


# ------------------------------------------------------------
# STEP 6: Dashboard layout — one section per room
# ------------------------------------------------------------
for room_id in ROOM_IDS:
    st.subheader(f"Room: {room_id}")

    readings = st.session_state.history[room_id]
    features = compute_features(readings)
    status = classify_occupancy(features)

    col1, col2 = st.columns([1, 3])

    with col1:
        if status == "Occupied":
            st.success(f"Status: {status}")
        else:
            st.error(f"Status: {status}")

        st.metric("Avg Power (W)", f"{features['mean_power']:.1f}")
        st.metric("Variance", f"{features['variance']:.1f}")
        st.metric("Peak Count", features["peak_count"])

    with col2:
        chart_df = pd.DataFrame({"power": readings})
        st.line_chart(chart_df["power"])

    st.divider()

# ------------------------------------------------------------
# STEP 7: Auto-refresh so the dashboard updates continuously
# ------------------------------------------------------------
st.caption("Dashboard auto-refreshes every 2 seconds (simulated live data).")
time.sleep(2)
st.rerun()

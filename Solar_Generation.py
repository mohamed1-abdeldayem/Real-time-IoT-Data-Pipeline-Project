import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import random
import time
from datetime import datetime
import csv
import os
from kafka import KafkaProducer
import json
from astral.sun import sun
from astral import LocationInfo
import pytz

# ===================== CONFIGURATION =====================
API_KEY = "440f752a883d097055e5b7bb9e7873c3"
FETCH_API_INTERVAL = 600   # Fetch new weather data every 10 min
UPDATE_INTERVAL = 5        # Update predictions every 5 sec
CSV_FILE = "solar_farm_data_log.csv"

# ---- Solar Constants ----
SOLAR_PANEL_AREA = 1.7
SOLAR_PANEL_EFFICIENCY = 0.18
SYSTEM_LOSS_FACTOR = 0.85

# ===================== KAFKA CONFIGURATION =====================
KAFKA_BROKER = "localhost:19092"
KAFKA_TOPIC = "solar-stations"

# إنشاء Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8")
)

# ===================== STATIONS =====================
stations = [
    {
        "id": "BSPP",
        "name": "Benban Solar Park",
        "type": "solar",
        "lat": 24.4560,
        "lon": 32.7390,
        "capacity_MW": 1650,
        "num_panels_est": 4125000
    },
    {
        "id": "KOSPP",
        "name": "Kom Ombo Solar Plant",
        "type": "solar",
        "lat": 24.6325,
        "lon": 32.8398,
        "capacity_MW": 200,
        "num_panels_est": 500000
    },
    {
        "id": "ZFSPP",
        "name": "Zafarana Solar Power Plant",
        "type": "solar",
        "lat": 29.2,
        "lon": 32.6,
        "capacity_MW": 25,
        "num_panels_est": 62500
    }
]

st.set_page_config(page_title="Solar Farms Real-time", layout="wide")
st.title("☀ Solar Farms — Real-time Power Monitoring")

# ===================== SOLAR POWER CALCULATIONS =====================

def get_weather(station):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={station['lat']}&lon={station['lon']}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            st.warning(f"API error for {station['name']}: {response.text}")
            return None

        data = response.json()
        main = data.get("main", {})
        clouds = data.get("clouds", {}).get("all", 0)

        temp = main.get("temp", 25)
        pressure = main.get("pressure", 1013)
        humidity = main.get("humidity", 50)
        timestamp = datetime.now(pytz.UTC)

        solar_irradiance = (1000 * (1 - clouds / 100))
        solar_irradiance = max(solar_irradiance, 50)

        return {
            "station_id": station["id"],
            "timestamp": timestamp,
            "temperature_C": temp,
            "pressure_hPa": pressure,
            "humidity_%": humidity,
            "clouds_%": clouds,
            "solar_irradiance_Wm2": round(solar_irradiance, 2),
            "data_source": "API"
        }

    except Exception as e:
        st.warning(f"⚠ Error fetching data for {station['name']}: {e}")
        return None


def is_daytime(station, timestamp):
    loc = LocationInfo(name=station["name"], region="Egypt", latitude=station["lat"], longitude=station["lon"])
    s = sun(loc.observer, date=timestamp.date(), tzinfo=pytz.UTC)
    return s['sunrise'] <= timestamp <= s['sunset']


def calculate_solar_power(station, weather_data):
    if not is_daytime(station, weather_data["timestamp"]):
        return {
            "temperature_C": weather_data["temperature_C"],
            "panel_temperature_C": weather_data["temperature_C"],
            "solar_irradiance_Wm2": 0,
            "effective_efficiency": 0,
            "power_kW": 0,
            "energy_kWh_10min": 0
        }

    irradiance = weather_data["solar_irradiance_Wm2"]
    temp = weather_data["temperature_C"]

    temp_coeff = -0.0045
    temp_loss = 1 + temp_coeff * (temp - 25)

    effective_efficiency = SOLAR_PANEL_EFFICIENCY * temp_loss
    effective_efficiency = max(effective_efficiency, 0.05)

    # درجة حرارة اللوح
    panel_temp = temp + random.uniform(3, 8)

    power_per_panel = irradiance * SOLAR_PANEL_AREA * effective_efficiency * SYSTEM_LOSS_FACTOR
    total_power_W = power_per_panel * station["num_panels_est"]
    total_power_kW = total_power_W / 1000
    energy_kWh_10min = total_power_kW * (10 / 60)

    return {
        "temperature_C": temp,
        "panel_temperature_C": round(panel_temp, 2),
        "solar_irradiance_Wm2": irradiance,
        "effective_efficiency": round(effective_efficiency, 3),
        "power_kW": round(total_power_kW, 2),
        "energy_kWh_10min": round(energy_kWh_10min, 2)
    }

# ===================== CSV + KAFKA PRODUCER =====================

def append_to_csv(data_dict):
    file_exists = os.path.isfile(CSV_FILE)
    fields = [
        "timestamp", "station_id", "data_source",
        "temperature_C", "panel_temperature_C",
        "solar_irradiance_Wm2", "effective_efficiency",
        "power_kW", "energy_kWh_10min"
    ]

    try:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)

        # إرسال البيانات إلى Kafka
        try:
            producer.send(KAFKA_TOPIC, data_dict)
            producer.flush()
        except Exception as e:
            st.warning(f"⚠ Could not send to Kafka: {e}")

    except Exception as e:
        st.warning(f"⚠ Could not write to CSV: {e}")


def generate_real_time_prediction(api_data, station):
    current_time = datetime.now(pytz.UTC)
    time_seed = int(current_time.timestamp() // 10)
    np.random.seed(time_seed)
    fluctuation_factor = 1 + np.random.uniform(-0.05, 0.05)

    predicted_irradiance = api_data["solar_irradiance_Wm2"] * fluctuation_factor
    predicted_irradiance = max(predicted_irradiance, 0)

    new_predictions = calculate_solar_power(station, {
        "solar_irradiance_Wm2": predicted_irradiance,
        "temperature_C": api_data["temperature_C"],
        "timestamp": current_time
    })

    prediction = {
        "timestamp": current_time,
        "station_id": station["id"],
        "data_source": "PREDICTION",
        **new_predictions
    }

    return prediction


# ===================== STREAMLIT DASHBOARD =====================

if "station_data" not in st.session_state:
    st.session_state.station_data = {s["id"]: pd.DataFrame() for s in stations}
    st.session_state.last_fetch_time = None
    st.session_state.last_update_time = None
    st.session_state.real_time_data = {s["id"]: [] for s in stations}

current_time = datetime.now(pytz.UTC)
should_fetch = False
if st.session_state.last_fetch_time is None:
    should_fetch = True
else:
    elapsed = (current_time - st.session_state.last_fetch_time).total_seconds()
    if elapsed >= FETCH_API_INTERVAL:
        should_fetch = True

if should_fetch:
    st.sidebar.info("☀ Fetching new solar data...")
    for station in stations:
        api_data = get_weather(station)
        if api_data:
            calc_data = calculate_solar_power(station, api_data)
            record = {**api_data, **calc_data}

            df = st.session_state.station_data[station["id"]]
            new_df = pd.DataFrame([record])
            st.session_state.station_data[station["id"]] = pd.concat([df, new_df], ignore_index=True)

            append_to_csv(record)

    st.session_state.last_fetch_time = current_time
    st.sidebar.success(f"✅ Data updated at {current_time.strftime('%H:%M:%S')}")

should_update = False
if st.session_state.last_update_time is None:
    should_update = True
else:
    elapsed = (current_time - st.session_state.last_update_time).total_seconds()
    if elapsed >= UPDATE_INTERVAL:
        should_update = True

if should_update:
    for station in stations:
        sid = station["id"]
        api_df = st.session_state.station_data[sid]

        if not api_df.empty:
            latest = api_df.iloc[-1].to_dict()
            pred = generate_real_time_prediction(latest, station)
            st.session_state.real_time_data[sid].append(pred)
            append_to_csv(pred)

    st.session_state.last_update_time = current_time


for station in stations:
    st.subheader(f"🔆 {station['name']} ({station['capacity_MW']} MW)")
    sid = station["id"]
    api_df = st.session_state.station_data[sid]
    rt_data = st.session_state.real_time_data[sid]

    if not api_df.empty and rt_data:
        current = rt_data[-1]
        latest_api = api_df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Power Output", f"{current['power_kW']:.0f} kW",
                      delta=f"{current['power_kW'] - latest_api['power_kW']:.0f} kW")
        with col2:
            st.metric("Solar Irradiance", f"{current['solar_irradiance_Wm2']:.0f} W/m²")
        with col3:
            st.metric("Temperature", f"{current['temperature_C']:.1f} °C")
        with col4:
            st.metric("Efficiency", f"{current['effective_efficiency']*100:.1f}%")

        rt_df = pd.DataFrame(rt_data).set_index("timestamp")
        if not rt_df.empty:
            st.line_chart(rt_df[["power_kW"]])
            st.caption("📈 Power Output over Time (kW)")

        if current["power_kW"] < 100:
            st.error(f"⚠ LOW OUTPUT ALERT: {current['power_kW']:.1f} kW is below safe range!")

    else:
        st.info("⏳ Waiting for first data fetch...")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛 Controls")

if st.sidebar.button("🔄 Force Refresh"):
    st.session_state.last_fetch_time = None
    st.session_state.last_update_time = None
    st.rerun()

if st.sidebar.button("📥 Download CSV"):
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.sidebar.download_button(
                label="Download Solar Data",
                data=f,
                file_name=CSV_FILE,
                mime="text/csv"
            )
    else:
        st.sidebar.warning("CSV not found yet.")

st.sidebar.markdown("---")
st.sidebar.info(f"Auto-refresh every {UPDATE_INTERVAL} seconds")
st_autorefresh(interval=UPDATE_INTERVAL * 1000, key="auto_refresh")

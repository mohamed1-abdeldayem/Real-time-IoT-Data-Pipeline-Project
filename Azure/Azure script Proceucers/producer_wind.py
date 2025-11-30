    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
    import pandas as pd
    import numpy as np
    import requests
    import random
    from scipy.interpolate import interp1d
    import time
    from datetime import datetime, timedelta
    import csv
    import os
    from azure.eventhub import EventHubProducerClient, EventData
    import json


    # ===================== SETTINGS =====================
    API_KEY = "440f752a883d097055e5b7bb9e7873c3"
    HUB_HEIGHT = 100
    REF_HEIGHT = 10
    ALPHA = 0.14
    TURBINE_DIAMETER = 82  # meters (typical large turbine)
    TURBINE_EFFICIENCY = 0.4  # Betz limit is ~0.59, practical is lower

    FETCH_API_INTERVAL = 600  # Fetch every 10 minutes
    UPDATE_INTERVAL = 5       # Update display every 5 seconds
    ALERT_THRESHOLD_KW = 5.0
    CSV_FILE = "wind_farm_data_log.csv"  # CSV file name


    # ===================== AZURE EVENT HUB SETTINGS =====================
    EVENT_HUB_CONNECTION_STR = os.getenv("EVENT_HUB_CONNECTION_STR")
    EVENT_HUB_NAME_WIND = "wind-data"

    producer_wind = EventHubProducerClient.from_connection_string(
        conn_str=EVENT_HUB_CONNECTION_STR,
        eventhub_name=EVENT_HUB_NAME_WIND
    )

    # ===================== WIND STATIONS =====================
    stations = [
        {"id": "WBWF", "name": "West Bakr Wind Farm", "lat": 28.531306, "lon": 32.823417, "num_turbines": 96},
        {"id": "GZWF", "name": "Gabal Elzeit Wind Farm", "lat": 29.2, "lon": 32.5, "num_turbines": 290},
        {"id": "ZFWF", "name": "Zafarana Wind Farm", "lat": 29.22, "lon": 33.6, "num_turbines": 50}
    ]

    # ===================== FUNCTIONS =====================
    def append_to_csv(data_dict):
        """Append data to CSV file"""
        file_exists = os.path.isfile(CSV_FILE)
        
        # Define the field order (same as your data structure)
        fields = [
            "timestamp", "station_id", "data_source",
            "wind_speed_mps", "wind_dir_deg", "air_temperature_C", 
            "air_pressure_hPa", "humidity_percent", "air_density_kgm3",
            "wind_speed_hub_mps", "turbine_power_kW", "farm_power_kW",
            "farm_energy_kWh_10min", "farm_energy_MWh_10min"
        ]
        
        try:
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                # Write the data row
                writer.writerow(data_dict)

            # Send to Azure Event Hub
            try:
                event_data = EventData(json.dumps(data_dict, default=str))
                batch = producer_wind.create_batch()
                batch.add(event_data)
                producer_wind.send_batch(batch)

            except Exception as e:
                st.warning(f"⚠ Could not send to Azure Event Hub: {e}")

        except Exception as e:
            st.warning(f"⚠ Could not write to CSV: {e}")

    def calculate_power_predictions(wind_speed_mps, wind_dir_deg, air_temp, air_pressure, humidity, num_turbines):
        """Calculate all power predictions from weather parameters"""
        # Adjust to hub height using wind profile power law
        wind_speed_hub_mps = wind_speed_mps * (HUB_HEIGHT / REF_HEIGHT) ** ALPHA

        # Air density calculation
        pressure_Pa = air_pressure * 100  # Convert hPa to Pa
        temp_K = air_temp + 273.15       # Convert Celsius to Kelvin
        R = 287.05  # Specific gas constant for dry air (J/kg·K)
        air_density = pressure_Pa / (R * temp_K)

        # Turbine power calculation
        swept_area = np.pi * (TURBINE_DIAMETER / 2) ** 2  # m²
        theoretical_power = 0.5 * air_density * swept_area * (wind_speed_hub_mps ** 3)  # Watts
        turbine_power_kW = theoretical_power * TURBINE_EFFICIENCY / 1000  # Convert to kW
        
        # Apply power curve correction (turbines don't produce power below cut-in or above cut-out)
        cut_in_speed = 3.0  # m/s
        rated_speed = 12.0  # m/s
        cut_out_speed = 25.0  # m/s
        rated_power = 2500  # kW (typical large turbine)
        
        if wind_speed_hub_mps < cut_in_speed or wind_speed_hub_mps > cut_out_speed:
            corrected_turbine_power = 0
        elif wind_speed_hub_mps > rated_speed:
            corrected_turbine_power = rated_power
        else:
            # Scale power between cut-in and rated speed
            corrected_turbine_power = min(turbine_power_kW, rated_power)
        
        # Farm calculations
        farm_power_kW = corrected_turbine_power * num_turbines
        farm_energy_kWh = farm_power_kW * (10 / 60)  # Energy for 10 minutes in kWh
        farm_energy_MWh_10min = farm_energy_kWh / 1000  # Convert to MWh

        return {
            "wind_speed_mps": round(wind_speed_mps, 2),
            "wind_dir_deg": wind_dir_deg,
            "air_temperature_C": air_temp,
            "air_pressure_hPa": air_pressure,
            "humidity_percent": humidity,
            "air_density_kgm3": round(air_density, 3),
            "wind_speed_hub_mps": round(wind_speed_hub_mps, 2),
            "turbine_power_kW": round(corrected_turbine_power, 2),
            "farm_power_kW": round(farm_power_kW, 2),
            "farm_energy_kWh_10min": round(farm_energy_kWh, 3),
            "farm_energy_MWh_10min": round(farm_energy_MWh_10min, 6)
        }

    def get_weather(station):
        """Fetch current weather data for a station and calculate power predictions"""
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={station['lat']}&lon={station['lon']}&appid={API_KEY}&units=metric"
        try:
            for _ in range(3):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        break
                except:
                    time.sleep(2)
            else:
                st.warning(f"API failed for {station['name']}")
                return None

            
            data = response.json()
            wind_data = data.get("wind", {})
            main_data = data.get("main", {})
            
            # Extract weather parameters from API
            wind_speed_mps = wind_data.get("speed", None)
            wind_dir_deg = wind_data.get("deg", random.randint(0, 360))
            air_temp = main_data.get("temp", 25)
            air_pressure = main_data.get("pressure", 1013)
            humidity = main_data.get("humidity", 50)
            timestamp = datetime.utcnow()

            # If no wind data, simulate realistic values
            if wind_speed_mps is None:
                st.warning(f"No wind data for {station['name']} — simulating.")
                wind_speed_mps = random.uniform(4, 10)
                wind_dir_deg = random.randint(0, 360)

            # Calculate all predictions
            predictions = calculate_power_predictions(
                wind_speed_mps, wind_dir_deg, air_temp, air_pressure, humidity, station["num_turbines"]
            )
            
            # Add metadata
            predictions.update({
                "station_id": station["id"],
                "timestamp": timestamp,
                "data_source": "API"  # Mark this as API data
            })
            
            return predictions

        except Exception as e:
            st.warning(f"⚠ Error fetching {station['name']}: {e}")
            return None

    def generate_real_time_prediction(api_data, station_id):
        """Generate smoothed real-time prediction using relative noise + diurnal & pattern factors."""
        current_time = datetime.utcnow()
        time_seed = int(current_time.timestamp() // 10)

        # pattern factor (station-specific)
        if station_id == "WBWF":
            pattern_factor = 1.0 + 0.15 * np.sin(time_seed * 0.1)
        elif station_id == "GZWF":
            pattern_factor = 1.0 + 0.10 * np.sin(time_seed * 0.2)
        else:  # ZFWF
            pattern_factor = 1.0 + 0.20 * np.sin(time_seed * 0.05)

        # diurnal factor (hour-of-day)
        hour = current_time.hour
        diurnal_factor = 0.8 + 0.4 * np.sin((hour - 6) * np.pi / 12)  # as before

        # relative noise (percentage) - smaller for low speeds
        base_wind = float(api_data.get("wind_speed_mps", 0.0))
        # scale noise amplitude by base_wind (so low wind => low absolute noise)
        if base_wind < 1.0:
            noise_pct = random.uniform(-0.05, 0.05)   # ±5%
        elif base_wind < 3.0:
            noise_pct = random.uniform(-0.08, 0.08)   # ±8%
        else:
            noise_pct = random.uniform(-0.12, 0.12)   # ±12%

        # compute raw predicted surface wind (relative model)
        raw_pred = base_wind * pattern_factor * diurnal_factor * (1.0 + noise_pct)

        # avoid negative or extremely small unrealistic negative values
        raw_pred = max(0.0, raw_pred)

        # smoothing: apply EWMA with previous real-time value if available
        prev_speed = None
        prev_list = st.session_state.real_time_data.get(station_id, [])
        if prev_list and isinstance(prev_list[-1].get("wind_speed_mps"), (int, float)):
            prev_speed = float(prev_list[-1]["wind_speed_mps"])
        # smoothing factor alpha (0..1) - higher alpha = follow new value more closely
        alpha = 0.35
        if prev_speed is not None:
            smoothed_surface = alpha * raw_pred + (1 - alpha) * prev_speed
        else:
            smoothed_surface = raw_pred

        # Ensure very small floor to avoid tiny floating noise (but don't force to cut-in)
        smoothed_surface = max(0.0, smoothed_surface)

        # Recalculate all parameters using the smoothed surface wind speed
        new_predictions = calculate_power_predictions(
            smoothed_surface,
            api_data.get("wind_dir_deg", 0),
            api_data.get("air_temperature_C", 25.0),
            api_data.get("air_pressure_hPa", 1013.0),
            api_data.get("humidity_percent", 50.0),
            next(s["num_turbines"] for s in stations if s["id"] == station_id)
        )

        # Build prediction dict (timestamp, data_source, and recalculated fields)
        prediction = api_data.copy()
        prediction.update(new_predictions)
        prediction["timestamp"] = current_time
        prediction["data_source"] = "PREDICTION"

        return prediction
    # ===================== STREAMLIT APP =====================
    st.set_page_config(page_title="Wind Farms Real-time", layout="wide")
    st.title("🌬 Wind Farms — Real-time Power Monitoring")

    # Initialize session state
    if "station_data" not in st.session_state:
        st.session_state.station_data = {s["id"]: pd.DataFrame() for s in stations}
        st.session_state.last_fetch_time = None
        st.session_state.last_update_time = None
        st.session_state.real_time_data = {s["id"]: [] for s in stations}

    # Get current time
    current_time = datetime.utcnow()

    # Check if we need to fetch new API data
    should_fetch = False
    if st.session_state.last_fetch_time is None:
        should_fetch = True
    else:
        time_since_fetch = (current_time - st.session_state.last_fetch_time).total_seconds()
        if time_since_fetch >= FETCH_API_INTERVAL:
            should_fetch = True

    # Fetch API data if needed
    if should_fetch:
        st.sidebar.info("🔄 Fetching new API data...")
        for station in stations:
            result = get_weather(station)
            if result:
                df = st.session_state.station_data[station["id"]]
                # Keep only last 6 hours of data
                six_hours_ago = current_time - timedelta(hours=6)
                if not df.empty:
                    df = df[df["timestamp"] >= six_hours_ago]
                new_df = pd.DataFrame([result])
                st.session_state.station_data[station["id"]] = pd.concat([df, new_df], ignore_index=True)
                
                # Initialize real-time data with current API value
                if not st.session_state.real_time_data[station["id"]]:
                    st.session_state.real_time_data[station["id"]] = [result]
                
                # ✅ ADDED: Log API data to CSV
                append_to_csv(result)
        
        st.session_state.last_fetch_time = current_time
        st.sidebar.success(f"✅ API data updated at {current_time.strftime('%H:%M:%S')}")

    # Update real-time data every 5 seconds
    if st.session_state.last_update_time is None:
        should_update = True
    else:
        time_since_update = (current_time - st.session_state.last_update_time).total_seconds()
        should_update = time_since_update >= UPDATE_INTERVAL

    if should_update:
        for station in stations:
            station_id = station["id"]
            api_data = st.session_state.station_data[station_id]
            
            if not api_data.empty:
                # Get latest API data as base
                latest_api_data = api_data.iloc[-1].to_dict()
                
                # Generate real-time prediction with SAME structure
                real_time_point = generate_real_time_prediction(latest_api_data, station_id)
                
                # Add to real-time data
                real_time_points = st.session_state.real_time_data[station_id]
                real_time_points.append(real_time_point)
                
                # Keep only recent data
                if len(real_time_points) > 100:
                    st.session_state.real_time_data[station_id] = real_time_points[-100:]
                
                # ✅ ADDED: Log real-time prediction to CSV
                append_to_csv(real_time_point)
        
        st.session_state.last_update_time = current_time

    # Display header with timing info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        next_fetch = FETCH_API_INTERVAL - (current_time - st.session_state.last_fetch_time).total_seconds() if st.session_state.last_fetch_time else 0
        st.metric("Next API Fetch", f"{max(0, int(next_fetch))}s")
    with col2:
        next_update = UPDATE_INTERVAL - (current_time - st.session_state.last_update_time).total_seconds() if st.session_state.last_update_time else 0
        st.metric("Next Update", f"{max(0, int(next_update))}s")
    with col3:
        st.metric("Current Time", current_time.strftime("%H:%M:%S"))
    with col4:
        total_data_points = sum(len(st.session_state.real_time_data[s["id"]]) for s in stations)
        st.metric("Total Data Points", total_data_points)

    # Display data flow info in sidebar
    st.sidebar.subheader("📡 Data Sources")
    st.sidebar.write("*API Data (10 min):*")
    st.sidebar.write("- Raw weather parameters")
    st.sidebar.write("- All predictions calculated")
    st.sidebar.write("*Real-time (5 sec):*") 
    st.sidebar.write("- Same parameters structure")
    st.sidebar.write("- Wind speed fluctuations")
    st.sidebar.write("- All values recalculated")

    # ✅ ADDED: CSV logging info
    st.sidebar.subheader("💾 Data Logging")
    st.sidebar.write(f"*CSV File:* {CSV_FILE}")
    st.sidebar.write("- Logs every API fetch")
    st.sidebar.write("- Logs every prediction")
    st.sidebar.write("- Complete data history")

    # Display real-time data for each station
    for station in stations:
        st.subheader(f"🏭 {station['name']} ({station['num_turbines']} turbines)")
        
        station_id = station["id"]
        api_data = st.session_state.station_data[station_id]
        real_time_data = st.session_state.real_time_data[station_id]
        
        if not api_data.empty and real_time_data:
            # Get current values
            latest_api = api_data.iloc[-1]
            current_real_time = real_time_data[-1]
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Farm Power", 
                    f"{current_real_time['farm_power_kW']:,.0f} kW",
                    delta=f"{(current_real_time['farm_power_kW'] - latest_api['farm_power_kW']):.0f} kW",
                    delta_color="normal"
                )
                st.metric("Data Source", current_real_time["data_source"])
            
            with col2:
                st.metric("Wind Speed", f"{current_real_time['wind_speed_mps']:.1f} m/s")
                st.metric("Wind Direction", f"{current_real_time['wind_dir_deg']}°")
            
            with col3:
                st.metric("Temperature", f"{current_real_time['air_temperature_C']:.1f} °C")
                st.metric("Air Density", f"{current_real_time['air_density_kgm3']:.3f} kg/m³")
            
            with col4:
                st.metric("10-min Energy", f"{current_real_time['farm_energy_kWh_10min']:,.0f} kWh")
                st.metric("Pressure", f"{current_real_time['air_pressure_hPa']:.1f} hPa")
            
            # Display real-time chart
            if len(real_time_data) > 1:
                # Convert to DataFrame for easier charting
                rt_df = pd.DataFrame(real_time_data)
                rt_df = rt_df.set_index("timestamp")
                
                # Show last 10 minutes of data
                ten_min_ago = current_time - timedelta(minutes=10)
                recent_data = rt_df[rt_df.index >= ten_min_ago]
                
                if not recent_data.empty:
                    st.subheader("📈 Real-time Power Trends")
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.line_chart(recent_data[['farm_power_kW']])
                        st.caption("Farm Power (kW) - Last 10 minutes")
                    
                    with chart_col2:
                        st.line_chart(recent_data[['wind_speed_mps', 'wind_speed_hub_mps']])
                        st.caption("Wind Speed (m/s) - Ground vs Hub Height")
                
                # Alert system
                if current_real_time['farm_power_kW'] < ALERT_THRESHOLD_KW:
                    st.error(f"🚨 LOW POWER ALERT: {current_real_time['farm_power_kW']:.1f} kW below {ALERT_THRESHOLD_KW} kW threshold!")
            
            # Show detailed parameters in expandable section
            with st.expander("📊 Detailed Parameters Comparison"):
                st.write("*All Parameters (API vs Real-time Prediction)*")
                
                # Create comparison table
                comparison_data = []
                params = [
                    "wind_speed_mps", "wind_dir_deg", "air_temperature_C", "air_pressure_hPa",
                    "humidity_percent", "air_density_kgm3", "wind_speed_hub_mps", 
                    "turbine_power_kW", "farm_power_kW", "farm_energy_kWh_10min", "farm_energy_MWh_10min"
                ]
                
                for param in params:
                    comparison_data.append({
                        "Parameter": param.replace("_", " ").title(),
                        "API Value": latest_api[param],
                        "Real-time Value": current_real_time[param],
                        "Difference": current_real_time[param] - latest_api[param] if isinstance(current_real_time[param], (int, float)) else "N/A"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Additional charts
                st.subheader("Historical Trends (Last 6 Hours)")
                if len(api_data) > 1:
                    historical_df = api_data.set_index("timestamp")
                    st.line_chart(historical_df[['farm_power_kW', 'wind_speed_mps']])
            
            # Station statistics
            with st.expander("📈 Station Statistics"):
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    if len(real_time_data) > 0:
                        avg_power = np.mean([rt['farm_power_kW'] for rt in real_time_data])
                        max_power = np.max([rt['farm_power_kW'] for rt in real_time_data])
                        st.metric("Avg Power (Recent)", f"{avg_power:,.0f} kW")
                        st.metric("Max Power (Recent)", f"{max_power:,.0f} kW")
                
                with stat_col2:
                    if len(api_data) > 0:
                        total_energy_mwh = api_data['farm_energy_MWh_10min'].sum()
                        avg_wind_speed = api_data['wind_speed_mps'].mean()
                        st.metric("Total Energy (6h)", f"{total_energy_mwh:.1f} MWh")
                        st.metric("Avg Wind Speed", f"{avg_wind_speed:.1f} m/s")
                
                with stat_col3:
                    st.metric("Turbine Count", station["num_turbines"])
                    st.metric("Data Points", len(real_time_data))
            
        else:
            st.warning("⏳ Waiting for initial data...")
            st.info("Data will appear after the first API fetch completes.")

    # System overview section
    st.sidebar.subheader("🔧 System Overview")
    st.sidebar.write(f"*Stations Monitoring:* {len(stations)}")
    st.sidebar.write(f"*API Interval:* {FETCH_API_INTERVAL // 60} min")
    st.sidebar.write(f"*Update Interval:* {UPDATE_INTERVAL} sec")
    st.sidebar.write(f"*Alert Threshold:* {ALERT_THRESHOLD_KW} kW")

    # Data statistics in sidebar
    st.sidebar.subheader("📊 Data Statistics")
    total_api_points = 0
    total_rt_points = 0

    for station in stations:
        station_id = station["id"]
        api_data = st.session_state.station_data[station_id]
        real_time_data = st.session_state.real_time_data[station_id]
        
        if not api_data.empty and real_time_data:
            current_data = real_time_data[-1]
            st.sidebar.write(f"{station['name']}:")
            st.sidebar.write(f"- Power: {current_data['farm_power_kW']:,.0f} kW")
            st.sidebar.write(f"- Wind: {current_data['wind_speed_mps']:.1f} m/s")
            st.sidebar.write(f"- Points: {len(real_time_data)}")
            st.sidebar.write("---")
            
            total_api_points += len(api_data)
            total_rt_points += len(real_time_data)

    st.sidebar.write(f"*Total API Points:* {total_api_points}")
    st.sidebar.write(f"*Total Real-time Points:* {total_rt_points}")

    # Control panel in sidebar
    st.sidebar.subheader("🎛 Controls")
    if st.sidebar.button("🔄 Force Refresh API Data"):
        st.session_state.last_fetch_time = None
        st.session_state.last_update_time = None
        st.sidebar.success("Forcing refresh...")
        st.rerun()

    # Download CSV button
    if st.sidebar.button("📥 Download CSV Data"):
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "rb") as f:
                st.sidebar.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=CSV_FILE,
                    mime="text/csv"
                )
        else:
            st.sidebar.warning("CSV file not created yet")

    # Adjustable settings
    st.sidebar.subheader("⚙ Settings")
    new_update_interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=1,
        max_value=30,
        value=UPDATE_INTERVAL,
        key="update_interval"
    )

    new_alert_threshold = st.sidebar.number_input(
        "Alert Threshold (kW)",
        min_value=0.0,
        max_value=1000.0,
        value=ALERT_THRESHOLD_KW,
        step=10.0,
        key="alert_threshold"
    )

    UPDATE_INTERVAL = new_update_interval
    ALERT_THRESHOLD_KW = new_alert_threshold

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🌬 *Wind Farm Monitoring System*  ")
    st.sidebar.markdown("Real-time power predictions  ")

    # Auto-refresh the page
    st.sidebar.info(f"🔄 Auto-refreshing every {UPDATE_INTERVAL} seconds...")
    st_autorefresh(interval=UPDATE_INTERVAL * 1000, key="auto_refresh")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Wind Station Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #00B4D8;
        padding-bottom: 20px;
    }
    h2 {
        color: #0077B6;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌬️ Wind Station Performance Dashboard")
st.markdown("**Real-time monitoring and analysis of wind power generation**")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4150/4150884.png", width=100)
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
    
    st.markdown("---")
    st.subheader("📊 Analysis Options")
    
    show_power = st.checkbox("Power Generation", value=True)
    show_wind = st.checkbox("Wind Analysis", value=True)
    show_environmental = st.checkbox("Environmental Conditions", value=True)
    show_turbine = st.checkbox("Turbine Performance", value=True)
    show_comparison = st.checkbox("Station Comparison", value=True)
    show_correlations = st.checkbox("Correlations & Relations", value=True)
    
    st.markdown("---")
    st.subheader("🎯 Filters")
    filter_valid = st.checkbox("Show only valid data", value=False)

# Function to detect column mappings
def detect_columns(df):
    """Detect and map column names to standard names"""
    column_mapping = {}
    
    patterns = {
        'timestamp': ['time', 'date', 'datetime', 'timestamp'],
        'station': ['station', 'device', 'turbine', 'farm', 'id'],
        'wind_speed': ['wind_spe', 'wind_speed', 'speed', 'ws'],
        'wind_direction': ['wind_dir', 'direction', 'dir', 'wd'],
        'air_temp': ['air_temp', 'temperature', 'temp'],
        'air_pressure': ['air_press', 'pressure', 'press'],
        'humidity': ['humidity', 'humid', 'rh'],
        'air_density': ['air_densit', 'density'],
        'turbine_power': ['turbine_p', 'turbine_power', 'power'],
        'farm_power': ['farm_pow', 'farm_power'],
        'farm_energy': ['farm_ene', 'farm_energy', 'energy'],
        'is_valid': ['is_valid', 'valid', 'status']
    }
    
    df_columns_lower = {col: col.lower().replace(' ', '_') for col in df.columns}
    
    for standard_name, variations in patterns.items():
        for col, col_lower in df_columns_lower.items():
            if any(var in col_lower for var in variations):
                column_mapping[standard_name] = col
                break
    
    return column_mapping

# Load and process data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        
        # Detect column mappings
        col_map = detect_columns(df)
        
        # Rename columns
        rename_dict = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename_dict)
        
        # Parse timestamp
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='10min')
        else:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='10min')
        
        # Convert numeric columns
        numeric_columns = ['wind_speed', 'wind_direction', 'air_temp', 'air_pressure', 
                          'humidity', 'air_density', 'turbine_power', 'farm_power', 'farm_energy']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        if 'wind_speed' in df.columns and 'turbine_power' in df.columns:
            df['power_coefficient'] = df['turbine_power'] / (df['wind_speed'] ** 3 + 0.001)
        
        if 'wind_speed' in df.columns:
            df['wind_class'] = pd.cut(df['wind_speed'], 
                                     bins=[0, 3, 7, 12, 20, 100],
                                     labels=['Calm', 'Light', 'Moderate', 'Strong', 'Gale'])
        
        return df
    else:
        # Generate sample wind farm data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='10min')
        np.random.seed(42)
        
        stations = ['WBWF', 'GZWF', 'ZFWF']
        dfs = []
        
        for station in stations:
            wind_speed = np.abs(np.random.normal(8, 3, len(dates)))
            wind_dir = np.random.uniform(0, 360, len(dates))
            air_temp = np.random.normal(22, 3, len(dates))
            air_press = np.random.normal(1015, 5, len(dates))
            humidity = np.random.uniform(20, 60, len(dates))
            air_density = 1.225 - (air_temp - 15) * 0.004
            
            # Power calculation based on wind speed (cubic relationship)
            turbine_power = np.where(wind_speed < 3, 0,
                           np.where(wind_speed > 25, 0,
                           np.minimum(wind_speed ** 3 * 0.01, 2.0)))
            
            farm_power = turbine_power * np.random.uniform(0.9, 1.1, len(dates))
            farm_energy = farm_power * 0.167  # 10 min intervals
            
            station_df = pd.DataFrame({
                'timestamp': dates,
                'station': station,
                'wind_speed': wind_speed,
                'wind_direction': wind_dir,
                'air_temp': air_temp,
                'air_pressure': air_press,
                'humidity': humidity,
                'air_density': air_density,
                'turbine_power': turbine_power,
                'farm_power': farm_power,
                'farm_energy': farm_energy,
                'is_valid': True
            })
            
            dfs.append(station_df)
        
        return pd.concat(dfs, ignore_index=True)

# Helper functions
def safe_sum(df, col_name):
    if col_name in df.columns:
        return df[col_name].sum()
    return 0

def safe_mean(df, col_name):
    if col_name in df.columns:
        return df[col_name].mean()
    return 0

def safe_max(df, col_name):
    if col_name in df.columns:
        return df[col_name].max()
    return 0

# Load data
df = load_data(uploaded_file)

if df is not None and len(df) > 0:
    
    # Filter valid data if requested
    if filter_valid and 'is_valid' in df.columns:
        df = df[df['is_valid'] == True]
        st.info(f"📊 Showing only valid data: {len(df)} records")
    
    # Check for multi-station data
    has_stations = 'station' in df.columns and df['station'].nunique() > 1
    
    if has_stations:
        stations = df['station'].unique()
        st.success(f"🌬️ Detected {len(stations)} wind stations: {', '.join(map(str, stations))}")
    
    # Display detected columns
    if uploaded_file is not None:
        with st.expander("🔍 Detected Columns"):
            available_cols = [col for col in df.columns if col != 'timestamp']
            st.write("**Available metrics:**", ", ".join(available_cols[:10]))
    
    # Overall metrics
    st.subheader("📊 Overall System Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_energy = safe_sum(df, 'farm_energy')
    avg_wind_speed = safe_mean(df, 'wind_speed')
    max_wind_speed = safe_max(df, 'wind_speed')
    avg_power = safe_mean(df, 'farm_power')
    max_power = safe_max(df, 'farm_power')
    avg_temp = safe_mean(df, 'air_temp')
    
    with col1:
        st.metric("⚡ Total Energy", f"{total_energy:.2f} kWh")
    with col2:
        st.metric("🌬️ Avg Wind Speed", f"{avg_wind_speed:.2f} m/s")
    with col3:
        st.metric("💨 Max Wind Speed", f"{max_wind_speed:.2f} m/s")
    with col4:
        st.metric("⚡ Avg Power", f"{avg_power:.2f} MW")
    with col5:
        st.metric("📈 Peak Power", f"{max_power:.2f} MW")
    with col6:
        st.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°C")
    
    st.markdown("---")
    
    # Wind Analysis
    if show_wind:
        st.subheader("🌬️ Wind Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'wind_speed' in df.columns:
                fig_wind_speed = go.Figure()
                
                if has_stations:
                    colors = ['#00B4D8', '#0077B6', '#03045E', '#90E0EF', '#CAF0F8']
                    for idx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        fig_wind_speed.add_trace(go.Scatter(
                            x=station_data['timestamp'],
                            y=station_data['wind_speed'],
                            mode='lines',
                            name=station,
                            line=dict(color=colors[idx % len(colors)], width=2)
                        ))
                else:
                    fig_wind_speed.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['wind_speed'],
                        mode='lines',
                        name='Wind Speed',
                        line=dict(color='#00B4D8', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 180, 216, 0.2)'
                    ))
                
                # Add operational range
                fig_wind_speed.add_hrect(y0=3, y1=25, 
                                        fillcolor="green", opacity=0.1,
                                        annotation_text="Operational Range", 
                                        annotation_position="top right")
                
                fig_wind_speed.update_layout(
                    title="Wind Speed Over Time",
                    xaxis_title="Time",
                    yaxis_title="Wind Speed (m/s)",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig_wind_speed, use_container_width=True)
        
        with col2:
            if 'wind_direction' in df.columns:
                # Wind rose diagram
                fig_wind_rose = go.Figure()
                
                if has_stations:
                    for idx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        fig_wind_rose.add_trace(go.Scatterpolar(
                            r=station_data['wind_speed'],
                            theta=station_data['wind_direction'],
                            mode='markers',
                            name=station,
                            marker=dict(size=5, color=colors[idx % len(colors)])
                        ))
                else:
                    fig_wind_rose.add_trace(go.Scatterpolar(
                        r=df['wind_speed'],
                        theta=df['wind_direction'],
                        mode='markers',
                        marker=dict(size=5, color=df['wind_speed'], 
                                   colorscale='Viridis', showscale=True,
                                   colorbar=dict(title="Speed (m/s)"))
                    ))
                
                fig_wind_rose.update_layout(
                    title="Wind Rose Diagram",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max_wind_speed * 1.1])
                    ),
                    height=400
                )
                
                st.plotly_chart(fig_wind_rose, use_container_width=True)
        
        # Wind distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if 'wind_speed' in df.columns:
                fig_wind_hist = go.Figure()
                
                if has_stations:
                    for idx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        fig_wind_hist.add_trace(go.Histogram(
                            x=station_data['wind_speed'],
                            name=station,
                            opacity=0.7,
                            marker_color=colors[idx % len(colors)]
                        ))
                else:
                    fig_wind_hist.add_trace(go.Histogram(
                        x=df['wind_speed'],
                        marker_color='#00B4D8',
                        name='Wind Speed'
                    ))
                
                fig_wind_hist.update_layout(
                    title="Wind Speed Distribution",
                    xaxis_title="Wind Speed (m/s)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=350
                )
                
                st.plotly_chart(fig_wind_hist, use_container_width=True)
        
        with col2:
            if 'wind_class' in df.columns:
                wind_class_counts = df['wind_class'].value_counts()
                
                fig_wind_class = go.Figure(data=[go.Pie(
                    labels=wind_class_counts.index,
                    values=wind_class_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#CAF0F8', '#90E0EF', '#00B4D8', '#0077B6', '#03045E'])
                )])
                
                fig_wind_class.update_layout(
                    title="Wind Classification Distribution",
                    height=350
                )
                
                st.plotly_chart(fig_wind_class, use_container_width=True)
    
    # Power Generation
    if show_power:
        st.subheader("⚡ Power Generation Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'farm_power' in df.columns:
                fig_power = go.Figure()
                
                if has_stations:
                    for idx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        fig_power.add_trace(go.Scatter(
                            x=station_data['timestamp'],
                            y=station_data['farm_power'],
                            mode='lines',
                            name=station,
                            line=dict(color=colors[idx % len(colors)], width=2)
                        ))
                else:
                    fig_power.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['farm_power'],
                        mode='lines',
                        name='Power',
                        line=dict(color='#0077B6', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 119, 182, 0.2)'
                    ))
                
                fig_power.update_layout(
                    title="Power Output Over Time",
                    xaxis_title="Time",
                    yaxis_title="Power (MW)",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig_power, use_container_width=True)
        
        with col2:
            if 'farm_energy' in df.columns:
                if has_stations:
                    station_energy = df.groupby('station')['farm_energy'].sum().reset_index()
                    
                    fig_energy_bar = go.Figure(data=[go.Bar(
                        x=station_energy['station'],
                        y=station_energy['farm_energy'],
                        marker_color=colors[:len(stations)],
                        text=station_energy['farm_energy'].round(2),
                        textposition='auto'
                    )])
                    
                    fig_energy_bar.update_layout(
                        title="Total Energy by Station",
                        xaxis_title="Station",
                        yaxis_title="Energy (kWh)",
                        height=400
                    )
                else:
                    hourly_energy = df.groupby(df['timestamp'].dt.hour)['farm_energy'].sum().reset_index()
                    
                    fig_energy_bar = go.Figure(data=[go.Bar(
                        x=hourly_energy['timestamp'],
                        y=hourly_energy['farm_energy'],
                        marker_color='#0077B6'
                    )])
                    
                    fig_energy_bar.update_layout(
                        title="Energy by Hour",
                        xaxis_title="Hour",
                        yaxis_title="Energy (kWh)",
                        height=400
                    )
                
                st.plotly_chart(fig_energy_bar, use_container_width=True)
    
    # Correlations
    if show_correlations:
        st.subheader("🔗 Wind-Power Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'wind_speed' in df.columns and 'farm_power' in df.columns:
                fig_power_curve = go.Figure()
                
                if has_stations:
                    for idx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        fig_power_curve.add_trace(go.Scatter(
                            x=station_data['wind_speed'],
                            y=station_data['farm_power'],
                            mode='markers',
                            name=station,
                            marker=dict(size=6, color=colors[idx % len(colors)], opacity=0.6)
                        ))
                else:
                    fig_power_curve.add_trace(go.Scatter(
                        x=df['wind_speed'],
                        y=df['farm_power'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=df['air_temp'] if 'air_temp' in df.columns else df['farm_power'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Temp (°C)")
                        )
                    ))
                
                # Add theoretical power curve
                wind_range = np.linspace(0, max_wind_speed, 100)
                theoretical_power = np.where(wind_range < 3, 0,
                                   np.where(wind_range > 25, 0,
                                   np.minimum(wind_range ** 3 * 0.01, max_power)))
                
                fig_power_curve.add_trace(go.Scatter(
                    x=wind_range,
                    y=theoretical_power,
                    mode='lines',
                    name='Theoretical',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                corr = df['wind_speed'].corr(df['farm_power'])
                
                fig_power_curve.update_layout(
                    title=f"Power Curve (Correlation: {corr:.3f})",
                    xaxis_title="Wind Speed (m/s)",
                    yaxis_title="Power (MW)",
                    plot_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig_power_curve, use_container_width=True)
        
        with col2:
            if 'air_density' in df.columns and 'farm_power' in df.columns:
                fig_density_power = go.Figure()
                
                fig_density_power.add_trace(go.Scatter(
                    x=df['air_density'],
                    y=df['farm_power'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=df['wind_speed'] if 'wind_speed' in df.columns else df['farm_power'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Wind Speed")
                    )
                ))
                
                corr = df['air_density'].corr(df['farm_power'])
                
                fig_density_power.update_layout(
                    title=f"Power vs Air Density (Corr: {corr:.3f})",
                    xaxis_title="Air Density (kg/m³)",
                    yaxis_title="Power (MW)",
                    plot_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig_density_power, use_container_width=True)
    
    # Environmental conditions
    if show_environmental:
        st.subheader("🌡️ Environmental Conditions")
        
        env_cols = ['air_temp', 'air_pressure', 'humidity']
        available_env = [col for col in env_cols if col in df.columns]
        
        if available_env:
            fig_env = make_subplots(
                rows=len(available_env), cols=1,
                subplot_titles=[col.replace('_', ' ').title() for col in available_env],
                vertical_spacing=0.1
            )
            
            colors_env = ['#E63946', '#06D6A0', '#4ECDC4']
            
            for idx, col in enumerate(available_env):
                if has_stations:
                    for sidx, station in enumerate(stations):
                        station_data = df[df['station'] == station]
                        show_legend = (idx == 0)
                        fig_env.add_trace(
                            go.Scatter(
                                x=station_data['timestamp'],
                                y=station_data[col],
                                mode='lines',
                                name=station,
                                line=dict(color=colors[sidx % len(colors)]),
                                showlegend=show_legend
                            ),
                            row=idx+1, col=1
                        )
                else:
                    fig_env.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[col],
                            mode='lines',
                            line=dict(color=colors_env[idx], width=2),
                            fill='tozeroy',
                            fillcolor=f'rgba{tuple(list(bytes.fromhex(colors_env[idx][1:])) + [0.2])}',
                            showlegend=False
                        ),
                        row=idx+1, col=1
                    )
            
            fig_env.update_xaxes(title_text="Time", row=len(available_env), col=1)
            
            fig_env.update_layout(
                height=300 * len(available_env),
                hovermode='x unified',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_env, use_container_width=True)
    
    # Station comparison
    if show_comparison and has_stations:
        st.subheader("📊 Station Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance comparison
            metrics_df = []
            for station in stations:
                station_data = df[df['station'] == station]
                metrics_df.append({
                    'Station': station,
                    'Total Energy (kWh)': safe_sum(station_data, 'farm_energy'),
                    'Avg Power (MW)': safe_mean(station_data, 'farm_power'),
                    'Avg Wind Speed (m/s)': safe_mean(station_data, 'wind_speed'),
                    'Capacity Factor (%)': (safe_mean(station_data, 'farm_power') / max_power * 100) if max_power > 0 else 0
                })
            
            metrics_df = pd.DataFrame(metrics_df)
            
            st.dataframe(
                metrics_df.style.background_gradient(cmap='Blues', subset=['Total Energy (kWh)', 'Avg Power (MW)']),
                use_container_width=True
            )
        
        with col2:
            # Inter-station correlation
            if 'farm_power' in df.columns and len(stations) >= 2:
                station_pivot = df.pivot_table(
                    index='timestamp',
                    columns='station',
                    values='farm_power'
                )
                
                if len(station_pivot) > 0:
                    station_corr = station_pivot.corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=station_corr.values,
                        x=station_corr.columns,
                        y=station_corr.columns,
                        colorscale='Blues',
                        text=station_corr.values.round(3),
                        texttemplate='%{text}',
                        textfont={"size": 12},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig_corr.update_layout(
                        title="Power Correlation Between Stations",
                        height=350
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="wind_station_data.csv",
            mime="text/csv"
        )
    
    # Summary statistics
    with st.expander("📊 Summary Statistics"):
        cols_to_show = [col for col in ['wind_speed', 'wind_direction', 'farm_power', 'air_temp'] if col in df.columns]
        
        if cols_to_show:
            for i in range(0, len(cols_to_show), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(cols_to_show):
                        st.write(f"**{cols_to_show[i].replace('_', ' ').title()}**")
                        st.write(df[cols_to_show[i]].describe())
                
                with col2:
                    if i + 1 < len(cols_to_show):
                        st.write(f"**{cols_to_show[i+1].replace('_', ' ').title()}**")
                        st.write(df[cols_to_show[i+1]].describe())

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format:
    Your CSV should include columns like:
    - **timestamp/date** - Time of measurement
    - **station/farm** - Station identifier
    - **wind_speed** - Wind speed (m/s)
    - **wind_direction** - Wind direction (degrees)
    - **air_temp** - Air temperature (°C)
    - **air_pressure** - Air pressure (hPa)
    - **humidity** - Relative humidity (%)
    - **turbine_power** - Turbine power output
    - **farm_power** - Farm power output
    - **farm_energy** - Energy generated
    
    The app automatically detects column names! 🎯
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌬️ Wind Station Dashboard | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)
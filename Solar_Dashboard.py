import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Solar Station Dashboard",
    page_icon="☀️",
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
        color: #FF6B35;
        padding-bottom: 20px;
    }
    h2 {
        color: #004E89;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("☀️ Solar Station Performance Dashboard")
st.markdown("**Multi-station monitoring and comparative analysis**")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4443/4443015.png", width=100)
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
    
    st.markdown("---")
    st.subheader("📊 Analysis Options")
    
    show_individual = st.checkbox("Individual Station Analysis", value=True)
    show_comparison = st.checkbox("Station Comparison", value=True)
    show_power = st.checkbox("Power Generation", value=True)
    show_voltage_current = st.checkbox("Voltage & Current", value=True)
    show_temperature = st.checkbox("Temperature Analysis", value=True)
    show_irradiance = st.checkbox("Solar Irradiance", value=True)
    show_correlations = st.checkbox("Correlations & Relations", value=True)

# Function to detect column mappings
def detect_columns(df):
    """Detect and map column names to standard names"""
    column_mapping = {}
    
    patterns = {
        'timestamp': ['time', 'date', 'datetime', 'timestamp'],
        'station': ['station', 'device', 'panel', 'unit', 'id'],
        'power_output': ['power', 'output', 'generation', 'kw', 'watt'],
        'solar_irradiance': ['irradiance', 'solar', 'radiation', 'ghi', 'w/m2'],
        'temperature': ['temp', 'temperature', 'celsius', 'panel_temp'],
        'voltage': ['voltage', 'volt', 'v'],
        'current': ['current', 'amp', 'ampere', 'a']
    }
    
    df_columns_lower = {col: col.lower() for col in df.columns}
    
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
        
        # Rename columns based on detected mappings
        rename_dict = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename_dict)
        
        # Try to parse timestamp column
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                st.warning("Could not parse timestamp column. Using index.")
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
        else:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
        
        # Convert numeric columns to proper types
        numeric_columns = ['power_output', 'solar_irradiance', 'temperature', 'voltage', 'current']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        if 'voltage' in df.columns and 'current' in df.columns:
            try:
                df['calculated_power'] = df['voltage'] * df['current'] / 1000  # kW
            except:
                pass
        
        return df
    else:
        # Generate sample multi-station data
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='15min')
        np.random.seed(42)
        
        stations = ['Station_A', 'Station_B', 'Station_C']
        dfs = []
        
        for station in stations:
            hours = dates.hour + dates.minute/60
            
            # Different performance characteristics per station
            if station == 'Station_A':
                efficiency_factor = 1.0
                temp_offset = 0
            elif station == 'Station_B':
                efficiency_factor = 0.95
                temp_offset = 3
            else:
                efficiency_factor = 0.90
                temp_offset = 5
            
            irradiance = np.maximum(0, 1000 * np.sin(np.pi * (hours - 6) / 12))
            irradiance = irradiance + np.random.normal(0, 50, len(dates))
            irradiance = np.maximum(0, irradiance)
            
            power = irradiance * 0.15 * efficiency_factor + np.random.normal(0, 5, len(dates))
            power = np.maximum(0, power)
            
            temperature = 20 + temp_offset + (irradiance / 100) * 2 + np.random.normal(0, 2, len(dates))
            
            voltage = 230 + np.random.normal(0, 5, len(dates))
            current = power * 1000 / voltage
            
            station_df = pd.DataFrame({
                'timestamp': dates,
                'station': station,
                'power_output': power,
                'solar_irradiance': irradiance,
                'temperature': temperature,
                'voltage': voltage,
                'current': current
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
    
    # Check if we have multi-station data
    has_stations = 'station' in df.columns and df['station'].nunique() > 1
    
    if has_stations:
        stations = df['station'].unique()
        st.info(f"📍 Detected {len(stations)} stations: {', '.join(map(str, stations))}")
    
    # Display detected columns
    if uploaded_file is not None:
        with st.expander("🔍 Detected Columns"):
            available_cols = [col for col in ['power_output', 'solar_irradiance', 'temperature', 'voltage', 'current', 'station'] if col in df.columns]
            st.write("**Available metrics:**", ", ".join(available_cols) if available_cols else "None detected")
    
    # Overall metrics
    st.subheader("📊 Overall System Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_power = safe_sum(df, 'power_output')
    avg_voltage = safe_mean(df, 'voltage')
    max_temp = safe_max(df, 'temperature')
    avg_irradiance = safe_mean(df, 'solar_irradiance')
    peak_power = safe_max(df, 'power_output')
    
    with col1:
        if 'power_output' in df.columns:
            st.metric(label="⚡ Total Energy", value=f"{total_power:.2f} kWh")
        else:
            st.metric(label="⚡ Total Energy", value="N/A")
    
    with col2:
        if 'voltage' in df.columns:
            st.metric(label="🔌 Avg Voltage", value=f"{avg_voltage:.1f} V")
        else:
            st.metric(label="🔌 Avg Voltage", value="N/A")
    
    with col3:
        if 'temperature' in df.columns:
            st.metric(label="🌡️ Peak Temp", value=f"{max_temp:.1f}°C")
        else:
            st.metric(label="🌡️ Peak Temp", value="N/A")
    
    with col4:
        if 'solar_irradiance' in df.columns:
            st.metric(label="☀️ Avg Irradiance", value=f"{avg_irradiance:.0f} W/m²")
        else:
            st.metric(label="☀️ Avg Irradiance", value="N/A")
    
    with col5:
        if 'power_output' in df.columns:
            st.metric(label="⚡ Peak Power", value=f"{peak_power:.2f} kW")
        else:
            st.metric(label="⚡ Peak Power", value="N/A")
    
    st.markdown("---")
    
    # STATION COMPARISON SECTION
    if has_stations and show_comparison:
        st.header("🔄 Station Comparison Analysis")
        
        # Station Performance Comparison Cards
        st.subheader("📈 Station Performance Overview")
        cols = st.columns(len(stations))
        
        for idx, station in enumerate(stations):
            station_data = df[df['station'] == station]
            with cols[idx]:
                st.markdown(f"### {station}")
                
                station_power = safe_sum(station_data, 'power_output')
                station_avg_temp = safe_mean(station_data, 'temperature')
                station_avg_voltage = safe_mean(station_data, 'voltage')
                
                st.metric("Total Energy", f"{station_power:.2f} kWh")
                st.metric("Avg Temp", f"{station_avg_temp:.1f}°C")
                st.metric("Avg Voltage", f"{station_avg_voltage:.1f} V")
        
        st.markdown("---")
        
        # Power Output Comparison
        if 'power_output' in df.columns:
            st.subheader("⚡ Power Output Comparison")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Time series comparison
                fig_power_comp = go.Figure()
                
                colors = ['#FF6B35', '#4ECDC4', '#F4A261', '#E63946', '#06D6A0']
                for idx, station in enumerate(stations):
                    station_data = df[df['station'] == station]
                    fig_power_comp.add_trace(go.Scatter(
                        x=station_data['timestamp'],
                        y=station_data['power_output'],
                        mode='lines',
                        name=station,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ))
                
                fig_power_comp.update_layout(
                    title="Power Output Comparison Over Time",
                    xaxis_title="Time",
                    yaxis_title="Power (kW)",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_power_comp, use_container_width=True)
            
            with col2:
                # Total energy by station
                station_totals = df.groupby('station')['power_output'].sum().reset_index()
                station_totals.columns = ['station', 'total_energy']
                
                fig_station_bar = go.Figure(data=[go.Bar(
                    x=station_totals['station'],
                    y=station_totals['total_energy'],
                    marker_color=colors[:len(stations)],
                    text=station_totals['total_energy'].round(2),
                    textposition='auto'
                )])
                
                fig_station_bar.update_layout(
                    title="Total Energy by Station",
                    xaxis_title="Station",
                    yaxis_title="Total Energy (kWh)",
                    height=400
                )
                
                st.plotly_chart(fig_station_bar, use_container_width=True)
        
        # Temperature Comparison
        if 'temperature' in df.columns:
            st.subheader("🌡️ Temperature Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_temp_comp = go.Figure()
                
                for idx, station in enumerate(stations):
                    station_data = df[df['station'] == station]
                    fig_temp_comp.add_trace(go.Scatter(
                        x=station_data['timestamp'],
                        y=station_data['temperature'],
                        mode='lines',
                        name=station,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ))
                
                fig_temp_comp.update_layout(
                    title="Temperature Comparison",
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=350
                )
                
                st.plotly_chart(fig_temp_comp, use_container_width=True)
            
            with col2:
                # Box plot for temperature distribution
                fig_temp_box = go.Figure()
                
                for idx, station in enumerate(stations):
                    station_data = df[df['station'] == station]
                    fig_temp_box.add_trace(go.Box(
                        y=station_data['temperature'],
                        name=station,
                        marker_color=colors[idx % len(colors)]
                    ))
                
                fig_temp_box.update_layout(
                    title="Temperature Distribution by Station",
                    yaxis_title="Temperature (°C)",
                    height=350
                )
                
                st.plotly_chart(fig_temp_box, use_container_width=True)
        
        # Inter-Station Correlation Analysis
        st.subheader("🔗 Inter-Station Relationships")
        
        if 'power_output' in df.columns:
            # Create pivot table for station power outputs
            station_pivot = df.pivot_table(
                index='timestamp',
                columns='station',
                values='power_output'
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Correlation heatmap between stations
                station_corr = station_pivot.corr()
                
                fig_station_corr = go.Figure(data=go.Heatmap(
                    z=station_corr.values,
                    x=station_corr.columns,
                    y=station_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=station_corr.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_station_corr.update_layout(
                    title="Power Output Correlation Between Stations",
                    height=400
                )
                
                st.plotly_chart(fig_station_corr, use_container_width=True)
            
            with col2:
                # Scatter matrix for first 2 stations
                if len(stations) >= 2:
                    station1, station2 = stations[0], stations[1]
                    
                    merged = station_pivot[[station1, station2]].dropna()
                    
                    if len(merged) > 0:
                        fig_scatter = go.Figure()
                        
                        # Create color array
                        color_array = list(range(len(merged)))
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=merged[station1],
                            y=merged[station2],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=color_array,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Time")
                            ),
                            text=[f"{s1:.2f} kW vs {s2:.2f} kW" for s1, s2 in zip(merged[station1], merged[station2])],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                        
                        # Add trend line
                        if len(merged) > 1:
                            z = np.polyfit(merged[station1], merged[station2], 1)
                            p = np.poly1d(z)
                            fig_scatter.add_trace(go.Scatter(
                                x=merged[station1],
                                y=p(merged[station1]),
                                mode='lines',
                                name='Trend',
                                line=dict(color='red', dash='dash', width=2)
                            ))
                        
                        corr = merged[station1].corr(merged[station2])
                        
                        fig_scatter.update_layout(
                            title=f"{station1} vs {station2} (Corr: {corr:.3f})",
                            xaxis_title=f"{station1} Power (kW)",
                            yaxis_title=f"{station2} Power (kW)",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Not enough data points for inter-station comparison")
        
        # Performance Ranking
        st.subheader("🏆 Station Performance Ranking")
        
        metrics_df = []
        for station in stations:
            station_data = df[df['station'] == station]
            metrics_df.append({
                'Station': station,
                'Total Energy (kWh)': safe_sum(station_data, 'power_output'),
                'Avg Power (kW)': safe_mean(station_data, 'power_output'),
                'Peak Power (kW)': safe_max(station_data, 'power_output'),
                'Avg Temperature (°C)': safe_mean(station_data, 'temperature'),
                'Avg Voltage (V)': safe_mean(station_data, 'voltage')
            })
        
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df = metrics_df.sort_values('Total Energy (kWh)', ascending=False)
        
        # Style the dataframe
        st.dataframe(
            metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Total Energy (kWh)', 'Avg Power (kW)']),
            use_container_width=True
        )
        
        st.markdown("---")
    
    # INDIVIDUAL STATION ANALYSIS
    if has_stations and show_individual:
        st.header("🔍 Individual Station Analysis")
        
        selected_station = st.selectbox("Select Station for Detailed Analysis", stations)
        station_df = df[df['station'] == selected_station]
        
        st.subheader(f"Analysis for {selected_station}")
        
        # Station metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Energy", f"{safe_sum(station_df, 'power_output'):.2f} kWh")
        with col2:
            st.metric("Avg Power", f"{safe_mean(station_df, 'power_output'):.2f} kW")
        with col3:
            st.metric("Avg Temp", f"{safe_mean(station_df, 'temperature'):.1f}°C")
        with col4:
            st.metric("Avg Voltage", f"{safe_mean(station_df, 'voltage'):.1f} V")
        
        # Detailed analysis for selected station
        if show_correlations:
            st.subheader(f"🔗 Correlations for {selected_station}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'power_output' in station_df.columns and 'solar_irradiance' in station_df.columns:
                    fig_corr1 = go.Figure()
                    fig_corr1.add_trace(go.Scatter(
                        x=station_df['solar_irradiance'],
                        y=station_df['power_output'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=station_df['temperature'] if 'temperature' in station_df.columns else station_df['power_output'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Temp (°C)")
                        )
                    ))
                    
                    z = np.polyfit(station_df['solar_irradiance'].dropna(), 
                                  station_df['power_output'].dropna(), 1)
                    p = np.poly1d(z)
                    fig_corr1.add_trace(go.Scatter(
                        x=station_df['solar_irradiance'],
                        y=p(station_df['solar_irradiance']),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    corr = station_df['solar_irradiance'].corr(station_df['power_output'])
                    
                    fig_corr1.update_layout(
                        title=f"Power vs Irradiance (Corr: {corr:.3f})",
                        xaxis_title="Solar Irradiance (W/m²)",
                        yaxis_title="Power (kW)",
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_corr1, use_container_width=True)
            
            with col2:
                if 'power_output' in station_df.columns and 'temperature' in station_df.columns:
                    fig_corr2 = go.Figure()
                    fig_corr2.add_trace(go.Scatter(
                        x=station_df['temperature'],
                        y=station_df['power_output'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=station_df['solar_irradiance'] if 'solar_irradiance' in station_df.columns else station_df['power_output'],
                            colorscale='YlOrRd',
                            showscale=True,
                            colorbar=dict(title="Irradiance")
                        )
                    ))
                    
                    corr = station_df['temperature'].corr(station_df['power_output'])
                    
                    fig_corr2.update_layout(
                        title=f"Power vs Temperature (Corr: {corr:.3f})",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Power (kW)",
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_corr2, use_container_width=True)
    
    # For non-station data or when stations not selected
    elif not has_stations:
        st.info("📍 Single station mode - upload multi-station data to see comparison features")
        
        # Show correlations for single station
        if show_correlations and 'power_output' in df.columns:
            st.subheader("🔗 Correlations & Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'solar_irradiance' in df.columns:
                    fig_corr1 = go.Figure()
                    fig_corr1.add_trace(go.Scatter(
                        x=df['solar_irradiance'],
                        y=df['power_output'],
                        mode='markers',
                        marker=dict(size=8, color='#FF6B35')
                    ))
                    
                    z = np.polyfit(df['solar_irradiance'].dropna(), df['power_output'].dropna(), 1)
                    p = np.poly1d(z)
                    fig_corr1.add_trace(go.Scatter(
                        x=df['solar_irradiance'],
                        y=p(df['solar_irradiance']),
                        mode='lines',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    corr = df['solar_irradiance'].corr(df['power_output'])
                    
                    fig_corr1.update_layout(
                        title=f"Power vs Irradiance (Corr: {corr:.3f})",
                        xaxis_title="Solar Irradiance (W/m²)",
                        yaxis_title="Power (kW)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_corr1, use_container_width=True)
            
            with col2:
                if 'temperature' in df.columns:
                    fig_corr2 = go.Figure()
                    fig_corr2.add_trace(go.Scatter(
                        x=df['temperature'],
                        y=df['power_output'],
                        mode='markers',
                        marker=dict(size=8, color='#E63946')
                    ))
                    
                    corr = df['temperature'].corr(df['power_output'])
                    
                    fig_corr2.update_layout(
                        title=f"Power vs Temperature (Corr: {corr:.3f})",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Power (kW)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_corr2, use_container_width=True)
    
    st.markdown("---")
    
    # Data Table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="solar_station_data.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format:
    Your CSV should include:
    - **timestamp/date** - Time of measurement
    - **station/device** - Station identifier (for multi-station analysis)
    - **power/output** - Power generation
    - **irradiance/solar** - Solar irradiance
    - **temperature/temp** - Panel temperature
    - **voltage/volt** - System voltage
    - **current/amp** - System current
    
    The app automatically detects column names and supports both single and multi-station data! 🎯
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌞 Solar Station Dashboard | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)
"""
⚙️ Settings - Configuration and System Management

Application configuration, data management, and system settings.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_loader import load_analytics_data, get_data_quality_summary

st.set_page_config(
    page_title="⚙️ Settings - Retail Analytics",
    page_icon="⚙️", 
    layout="wide"
)

st.title("⚙️ System Settings")
st.markdown("**Configuration and system management**")

# Initialize session state for settings
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'data_refresh_interval': 30,
        'default_forecast_horizon': 90,
        'cache_retention_days': 7,
        'auto_validation': True,
        'notification_email': '',
        'api_timeout': 30,
        'max_file_size_mb': 100
    }

# Sidebar navigation
st.sidebar.header("Settings Categories")
setting_category = st.sidebar.selectbox(
    "Select Category",
    ["Data Management", "Forecasting", "Validation", "VPP Analysis", "System", "Export/Import"]
)

# Data Management Settings
if setting_category == "Data Management":
    st.subheader("📊 Data Management Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Loading Configuration**")
        
        data_refresh_interval = st.number_input(
            "Auto-refresh Interval (minutes)",
            min_value=1,
            max_value=1440,
            value=st.session_state.settings['data_refresh_interval'],
            help="How often to automatically refresh data"
        )
        
        cache_retention_days = st.number_input(
            "Cache Retention (days)", 
            min_value=1,
            max_value=30,
            value=st.session_state.settings['cache_retention_days'],
            help="How long to keep cached data"
        )
        
        max_file_size_mb = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=1000,
            value=st.session_state.settings['max_file_size_mb'],
            help="Maximum file size for uploads"
        )
        
        if st.button("🔄 Clear All Cache"):
            try:
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
    
    with col2:
        st.markdown("**Data Quality Settings**")
        
        min_data_coverage = st.slider(
            "Minimum Data Coverage (%)",
            0, 100, 80,
            help="Minimum data coverage required for analysis"
        )
        
        gap_fill_method = st.selectbox(
            "Gap Fill Method",
            ["Linear Interpolation", "Forward Fill", "Backward Fill", "Mean Fill"],
            help="Method for filling data gaps"
        )
        
        outlier_detection = st.checkbox(
            "Enable Outlier Detection",
            value=True,
            help="Automatically detect and flag outliers"
        )
        
        outlier_threshold = st.slider(
            "Outlier Threshold (std dev)",
            1.0, 5.0, 3.0, 0.1,
            help="Standard deviation threshold for outlier detection"
        )
    
    # Data Quality Dashboard
    st.markdown("**Current Data Status**")
    
    try:
        df = load_analytics_data()
        quality_summary = get_data_quality_summary()
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                latest_date = df['DATE'].max() if len(df) > 0 else "No data"
                st.metric("Latest Data", str(latest_date)[:10])
            with col3:
                customer_count = df['CUSTOMER_NAME'].nunique()
                st.metric("Unique Customers", customer_count)
            with col4:
                data_coverage = (len(df.dropna()) / len(df) * 100) if len(df) > 0 else 0
                st.metric("Data Coverage", f"{data_coverage:.1f}%")
        
        if quality_summary:
            st.markdown("**Data Quality Details**")
            quality_df = pd.DataFrame(quality_summary)
            st.dataframe(quality_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Unable to load data status: {e}")

# Forecasting Settings
elif setting_category == "Forecasting":
    st.subheader("📈 Forecasting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Default Forecast Parameters**")
        
        default_horizon = st.number_input(
            "Default Forecast Horizon (days)",
            min_value=1,
            max_value=365,
            value=st.session_state.settings['default_forecast_horizon'],
            help="Default number of days to forecast"
        )
        
        confidence_levels = st.multiselect(
            "Confidence Intervals",
            [10, 20, 30, 40, 50, 60, 70, 80, 90],
            default=[10, 50, 90],
            help="Confidence levels for forecast intervals"
        )
        
        forecast_frequency = st.selectbox(
            "Forecast Frequency",
            ["Daily", "Hourly", "Half-hourly"],
            help="Time resolution for forecasts"
        )
        
        ensemble_models = st.multiselect(
            "Ensemble Models",
            ["Random Forest", "Gradient Boosting", "LSTM", "ARIMA"],
            default=["Random Forest", "Gradient Boosting"],
            help="Models to include in ensemble"
        )
    
    with col2:
        st.markdown("**Weather Integration**")
        
        weather_sources = st.multiselect(
            "Weather Data Sources",
            ["BOM", "WeatherAPI", "OpenWeather", "Local Station"],
            default=["BOM"],
            help="Weather data sources to use"
        )
        
        weather_lag_days = st.number_input(
            "Weather Forecast Lag (days)",
            min_value=0,
            max_value=14,
            value=1,
            help="Days of weather forecast lag to account for"
        )
        
        temperature_normalization = st.checkbox(
            "Enable Temperature Normalization",
            value=True,
            help="Normalize forecasts for temperature effects"
        )
        
        st.markdown("**Model Retraining**")
        
        retrain_frequency = st.selectbox(
            "Retrain Frequency",
            ["Weekly", "Monthly", "Quarterly", "Manual"],
            index=1,
            help="How often to retrain models"
        )
        
        min_training_data_months = st.number_input(
            "Min Training Data (months)",
            min_value=1,
            max_value=60,
            value=12,
            help="Minimum months of data required for training"
        )
    
    # Model Status
    st.markdown("**Current Model Status**")
    
    try:
        models_path = Path(__file__).parent.parent.parent / "models"
        if models_path.exists():
            model_files = list(models_path.glob("*.joblib")) + list(models_path.glob("*.h5"))
            
            if model_files:
                model_info = []
                for model_file in model_files:
                    stat = model_file.stat()
                    model_info.append({
                        'Model': model_file.stem,
                        'Size (MB)': stat.st_size / (1024 * 1024),
                        'Last Modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
                
                model_df = pd.DataFrame(model_info)
                st.dataframe(model_df, use_container_width=True)
            else:
                st.info("No trained models found")
        else:
            st.warning("Models directory not found")
            
    except Exception as e:
        st.error(f"Error checking model status: {e}")

# Validation Settings
elif setting_category == "Validation":
    st.subheader("✅ Validation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Validation Parameters**")
        
        auto_validation = st.checkbox(
            "Enable Auto-Validation",
            value=st.session_state.settings['auto_validation'],
            help="Automatically run validation after forecasting"
        )
        
        validation_split = st.slider(
            "Validation Split (%)",
            10, 50, 20,
            help="Percentage of data to use for validation"
        )
        
        accuracy_thresholds = st.subheader("Accuracy Thresholds")
        
        mape_threshold = st.number_input(
            "MAPE Warning Threshold (%)",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            help="MAPE threshold for warnings"
        )
        
        rmse_threshold = st.number_input(
            "RMSE Warning Threshold",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            help="RMSE threshold for warnings"
        )
    
    with col2:
        st.markdown("**Validation Reports**") 
        
        report_frequency = st.selectbox(
            "Report Frequency",
            ["Daily", "Weekly", "Monthly", "On-demand"],
            index=1,
            help="How often to generate validation reports"
        )
        
        report_recipients = st.text_area(
            "Report Recipients (emails)",
            placeholder="email1@company.com, email2@company.com",
            help="Comma-separated list of email addresses"
        )
        
        include_charts = st.checkbox(
            "Include Charts in Reports",
            value=True,
            help="Include visualizations in validation reports"
        )
        
        st.markdown("**Alert Settings**")
        
        enable_alerts = st.checkbox(
            "Enable Performance Alerts",
            value=True,
            help="Send alerts when accuracy drops below thresholds"
        )
        
        alert_cooldown = st.number_input(
            "Alert Cooldown (hours)",
            min_value=1,
            max_value=168,
            value=24,
            help="Minimum time between alerts"
        )

# VPP Analysis Settings
elif setting_category == "VPP Analysis":
    st.subheader("🔋 VPP Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Default VPP Parameters**")
        
        default_battery_capacity = st.number_input(
            "Default Battery Capacity (MWh)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            help="Default battery capacity for analysis"
        )
        
        default_battery_power = st.number_input(
            "Default Battery Power (MW)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            help="Default battery power rating"
        )
        
        default_efficiency = st.slider(
            "Default Round-trip Efficiency (%)",
            70, 95, 85,
            help="Default battery round-trip efficiency"
        )
        
        default_lifetime = st.number_input(
            "Default Project Lifetime (years)",
            min_value=5,
            max_value=25,
            value=15,
            help="Default project lifetime for analysis"
        )
    
    with col2:
        st.markdown("**Economic Assumptions**")
        
        default_capex = st.number_input(
            "Default CAPEX ($/MWh)",
            min_value=100000,
            max_value=1000000,
            value=400000,
            step=10000,
            help="Default capital expenditure per MWh"
        )
        
        default_opex = st.number_input(
            "Default OPEX ($/MW/year)",
            min_value=5000,
            max_value=50000,
            value=15000,
            step=1000,
            help="Default operational expenditure per MW per year"
        )
        
        default_discount_rate = st.slider(
            "Default Discount Rate (%)",
            3.0, 15.0, 7.0, 0.5,
            help="Default discount rate for NPV calculations"
        )
        
        revenue_streams = st.multiselect(
            "Default Revenue Streams",
            ["Energy Arbitrage", "FCAS Services", "Capacity Market", "Network Services"],
            default=["Energy Arbitrage", "FCAS Services", "Network Services"],
            help="Default revenue streams to include in analysis"
        )

# System Settings
elif setting_category == "System":
    st.subheader("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Application Settings**")
        
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            help="Application theme"
        )
        
        page_width = st.selectbox(
            "Page Layout",
            ["Wide", "Centered"],
            help="Page layout mode"
        )
        
        show_tooltips = st.checkbox(
            "Show Tooltips",
            value=True,
            help="Show helpful tooltips throughout the app"
        )
        
        api_timeout = st.number_input(
            "API Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.settings['api_timeout'],
            help="Timeout for API requests"
        )
    
    with col2:
        st.markdown("**Notification Settings**")
        
        notification_email = st.text_input(
            "Notification Email",
            value=st.session_state.settings['notification_email'],
            help="Email address for system notifications"
        )
        
        enable_email_notifications = st.checkbox(
            "Enable Email Notifications",
            value=True,
            help="Send email notifications for important events"
        )
        
        notification_types = st.multiselect(
            "Notification Types",
            ["Data Updates", "Forecast Complete", "Validation Alerts", "System Errors"],
            default=["Validation Alerts", "System Errors"],
            help="Types of notifications to receive"
        )
    
    # System Status
    st.markdown("**System Status**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("App Version", "v1.0.0")
    with col2:
        st.metric("Python Version", f"{sys.version.split()[0]}")
    with col3:
        uptime = "Running"  # Simplified
        st.metric("Status", uptime)
    with col4:
        st.metric("Cache Size", "~50MB")  # Simplified

# Export/Import Settings
elif setting_category == "Export/Import":
    st.subheader("📤 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Settings**")
        
        default_export_format = st.selectbox(
            "Default Export Format",
            ["CSV", "Parquet", "Excel", "JSON"],
            help="Default format for data exports"
        )
        
        include_metadata = st.checkbox(
            "Include Metadata in Exports",
            value=True,
            help="Include metadata (timestamps, settings) in exports"
        )
        
        compress_exports = st.checkbox(
            "Compress Exports",
            value=True,
            help="Compress export files to save space"
        )
        
        # Export current settings
        if st.button("📤 Export Current Settings"):
            settings_json = json.dumps(st.session_state.settings, indent=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📄 Download Settings JSON",
                data=settings_json,
                file_name=f"retail_analytics_settings_{timestamp}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**Import Settings**")
        
        # Import settings
        uploaded_settings = st.file_uploader(
            "Upload Settings File",
            type=['json'],
            help="Import settings from a JSON file"
        )
        
        if uploaded_settings is not None:
            try:
                imported_settings = json.load(uploaded_settings)
                
                st.json(imported_settings)
                
                if st.button("🔄 Apply Imported Settings"):
                    st.session_state.settings.update(imported_settings)
                    st.success("Settings imported successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error importing settings: {e}")
        
        # Reset settings
        st.markdown("**Reset Options**")
        
        if st.button("🔄 Reset to Defaults", type="secondary"):
            if st.checkbox("Confirm reset to defaults"):
                st.session_state.settings = {
                    'data_refresh_interval': 30,
                    'default_forecast_horizon': 90,
                    'cache_retention_days': 7,
                    'auto_validation': True,
                    'notification_email': '',
                    'api_timeout': 30,
                    'max_file_size_mb': 100
                }
                st.success("Settings reset to defaults!")
                st.rerun()

# Save settings button (global)
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("💾 Save Settings", type="primary"):
        # Update session state with current values
        if setting_category == "Data Management":
            st.session_state.settings.update({
                'data_refresh_interval': data_refresh_interval,
                'cache_retention_days': cache_retention_days,
                'max_file_size_mb': max_file_size_mb
            })
        elif setting_category == "Forecasting":
            st.session_state.settings.update({
                'default_forecast_horizon': default_horizon
            })
        elif setting_category == "Validation":
            st.session_state.settings.update({
                'auto_validation': auto_validation
            })
        elif setting_category == "System":
            st.session_state.settings.update({
                'notification_email': notification_email,
                'api_timeout': api_timeout
            })
        
        st.success("Settings saved successfully!")

st.markdown("---")
st.markdown("*Settings last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
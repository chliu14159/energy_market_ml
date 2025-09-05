import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_kpi_cards(df):
    """Create KPI cards for dashboard overview"""
    current_month = datetime.now().replace(day=1)
    current_month_data = df[pd.to_datetime(df['date']).dt.to_period('M') == current_month.strftime('%Y-%m')]
    
    # Handle different column names for load data
    load_col = None
    for col in ['load_mw', 'NET_CONSUMPTION_MW', 'HALFHOURLY_TOTAL_MW_E1', 'DAILY_TOTAL_MW']:
        if col in df.columns:
            load_col = col
            break
    
    if load_col is None:
        st.error("No load data column found")
        return
    
    total_load = df[load_col].sum()
    avg_load = df[load_col].mean()
    peak_load = df[load_col].max()
    current_month_load = current_month_data[load_col].sum() if not current_month_data.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Load (MWh)",
            value=f"{total_load:,.0f}",
            delta=f"{current_month_load:,.0f} this month"
        )
    
    with col2:
        st.metric(
            label="Average Load (MW)",
            value=f"{avg_load:.1f}",
            delta=f"{current_month_data[load_col].mean():.1f} avg this month" if not current_month_data.empty else "0.0 avg this month"
        )
    
    with col3:
        st.metric(
            label="Peak Load (MW)",
            value=f"{peak_load:.1f}",
            delta=f"{current_month_data[load_col].max():.1f} peak this month" if not current_month_data.empty else "0.0 peak this month"
        )
    
    with col4:
        data_coverage = len(df['date'].unique())
        st.metric(
            label="Data Coverage (Days)",
            value=f"{data_coverage}",
            delta="Historical data available"
        )

def create_load_trend_chart(df, title="Load Trend Analysis"):
    """Create load trend chart with moving averages"""
    df_copy = df.copy()
    
    # Handle different date column names
    date_col = None
    if 'date' in df_copy.columns:
        date_col = 'date'
    elif 'DATE' in df_copy.columns:
        date_col = 'DATE'
        df_copy['date'] = pd.to_datetime(df_copy['DATE'])
    elif 'DATE_TIME_HH' in df_copy.columns:
        date_col = 'DATE_TIME_HH'
        df_copy['date'] = pd.to_datetime(df_copy['DATE_TIME_HH'])
    else:
        # Create a simple index-based date
        df_copy['date'] = pd.date_range(start='2023-01-01', periods=len(df_copy), freq='30min')
    
    if date_col != 'date':
        df_copy['date'] = pd.to_datetime(df_copy[date_col])
    
    df_copy = df_copy.sort_values('date')
    
    # Handle different load column names
    load_col = None
    if 'load_mw' in df_copy.columns:
        load_col = 'load_mw'
    elif 'DAILY_TOTAL_MW' in df_copy.columns:
        load_col = 'DAILY_TOTAL_MW'
    elif 'NET_CONSUMPTION_MW' in df_copy.columns:
        load_col = 'NET_CONSUMPTION_MW'
    else:
        # Use first numeric column
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            load_col = numeric_cols[0]
        else:
            raise ValueError("No suitable load column found")
    
    # Calculate moving averages
    df_copy['ma_7'] = df_copy[load_col].rolling(window=7).mean()
    df_copy['ma_30'] = df_copy[load_col].rolling(window=30).mean()
    
    fig = go.Figure()
    
    # Add actual load
    fig.add_trace(go.Scatter(
        x=df_copy['date'],
        y=df_copy[load_col],
        mode='lines',
        name='Actual Load',
        line=dict(color='blue', width=1),
        opacity=0.7
    ))
    
    # Add 7-day moving average
    fig.add_trace(go.Scatter(
        x=df_copy['date'],
        y=df_copy['ma_7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='orange', width=2)
    ))
    
    # Add 30-day moving average
    fig.add_trace(go.Scatter(
        x=df_copy['date'],
        y=df_copy['ma_30'],
        mode='lines',
        name='30-Day MA',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_load_duration_curve(df, title="Load Duration Curve"):
    """Create load duration curve - handles both Series and DataFrame inputs"""
    # Handle both Series and DataFrame inputs
    if isinstance(df, pd.Series):
        # If it's a Series, use it directly as the load data
        load_data = df.dropna().sort_values(ascending=False)
    else:
        # Handle different column names for load data in DataFrame
        load_col = None
        for col in ['load_mw', 'NET_CONSUMPTION_MW', 'HALFHOURLY_TOTAL_MW_E1', 'DAILY_TOTAL_MW']:
            if col in df.columns:
                load_col = col
                break
        
        if load_col is None:
            raise ValueError("No load column found in dataframe")
        
        load_data = df[load_col].dropna().sort_values(ascending=False)
    
    # Create percentile positions (0 = highest load, 100 = lowest load)
    n_points = len(load_data)
    percentiles = np.linspace(0, 100, n_points)
    sorted_loads = load_data.reset_index(drop=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=sorted_loads,
        mode='lines',
        name='Load Duration Curve',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 123, 255, 0.1)'
    ))
    
    # Add POE (Probability of Exceedance) markers
    key_percentiles = [10, 25, 50, 75, 90]
    for p in key_percentiles:
        # POE calculation: POE10 means 10% of time load is AT OR ABOVE this value
        idx = int((p / 100) * (n_points - 1))
        if idx < len(sorted_loads):
            fig.add_trace(go.Scatter(
                x=[p],
                y=[sorted_loads.iloc[idx]],
                mode='markers+text',
                name=f'POE{p}',
                text=f'POE{p}: {sorted_loads.iloc[idx]:.1f}MW',
                textposition='top center',
                marker=dict(size=8, color='red')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Probability of Exceedance (% of time load exceeded)",
        yaxis_title="Load (MW)",
        height=400,
        showlegend=True,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def create_seasonal_pattern_chart(df, title="Seasonal Load Patterns"):
    """Create seasonal pattern analysis"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day_of_week'] = df_copy['date'].dt.day_name()
    df_copy['hour'] = df_copy['date'].dt.hour
    
    # Handle different column names for load data
    load_col = None
    for col in ['load_mw', 'NET_CONSUMPTION_MW', 'HALFHOURLY_TOTAL_MW_E1', 'DAILY_TOTAL_MW']:
        if col in df.columns:
            load_col = col
            break
    
    if load_col is None:
        st.error("No load data column found for seasonal patterns")
        return go.Figure()
    
    # Monthly patterns
    monthly_avg = df_copy.groupby('month')[load_col].mean().reset_index()
    monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%b')
    
    # Day of week patterns
    dow_avg = df_copy.groupby('day_of_week')[load_col].mean().reset_index()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg['day_of_week'] = pd.Categorical(dow_avg['day_of_week'], categories=dow_order, ordered=True)
    dow_avg = dow_avg.sort_values('day_of_week')
    
    # Hourly patterns
    hourly_avg = df_copy.groupby('hour')[load_col].mean().reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Monthly Average Load', 'Day of Week Average Load', 
                       'Hourly Average Load', 'Load Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Monthly pattern
    fig.add_trace(
        go.Bar(x=monthly_avg['month_name'], y=monthly_avg[load_col],
               name='Monthly Avg', marker_color='blue'),
        row=1, col=1
    )
    
    # Day of week pattern
    fig.add_trace(
        go.Bar(x=dow_avg['day_of_week'], y=dow_avg[load_col],
               name='DOW Avg', marker_color='green'),
        row=1, col=2
    )
    
    # Hourly pattern
    fig.add_trace(
        go.Scatter(x=hourly_avg['hour'], y=hourly_avg[load_col],
                  mode='lines+markers', name='Hourly Avg', line_color='orange'),
        row=2, col=1
    )
    
    # Load distribution histogram
    fig.add_trace(
        go.Histogram(x=df_copy[load_col], nbinsx=30,
                    name='Load Distribution', marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Load (MW)", row=2, col=2)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Load (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Load (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

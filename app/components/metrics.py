"""
Metrics Component

Calculates key performance indicators and portfolio metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_portfolio_metrics(df):
    """
    Calculate portfolio-level metrics
    
    Args:
        df (pd.DataFrame): Analytics dataframe
    
    Returns:
        dict: Portfolio metrics
    """
    if df is None or df.empty:
        return {}
    
    # Regional data (equivalent to customer data)
    region_data = df[df['REGIONID'].notna()]
    
    # Weather data
    weather_data = df[df['AIR_TEMP'].notna()]
    
    # Trading data
    trading_data = df[df['RRP'].notna()]
    
    metrics = {
        'total_regions': region_data['REGIONID'].nunique() if not region_data.empty else 0,
        'total_records': len(df),
        'region_records': len(region_data),
        'weather_records': len(weather_data),
        'trading_records': len(trading_data)
    }
    
    # Load metrics
    if not region_data.empty:
        # Use TOTALDEMAND as the main load metric
        if 'TOTALDEMAND' in region_data.columns:
            daily_load = region_data.groupby('DATE')['TOTALDEMAND'].mean()
        else:
            daily_load = pd.Series([0])
        
        metrics.update({
            'avg_daily_load_mw': daily_load.mean(),
            'max_daily_load_mw': daily_load.max(),
            'min_daily_load_mw': daily_load.min(),
            'load_volatility': daily_load.std()
        })
    
    # Price metrics
    if not trading_data.empty and 'RRP' in trading_data.columns:
        daily_price = trading_data.groupby('DATE')['RRP'].mean()
        
        metrics.update({
            'avg_daily_price': daily_price.mean(),
            'max_daily_price': daily_price.max(),
            'min_daily_price': daily_price.min(),
            'price_volatility': daily_price.std()
        })
    
    # Weather metrics
    if not weather_data.empty and 'AIR_TEMP' in weather_data.columns:
        daily_temp = weather_data.groupby('DATE')['AIR_TEMP'].mean()
        
        metrics.update({
            'avg_temperature': daily_temp.mean(),
            'max_temperature': daily_temp.max(),
            'min_temperature': daily_temp.min()
        })
    
    return metrics

def calculate_customer_metrics(df, region_id=None):
    """
    Calculate region-specific metrics
    
    Args:
        df (pd.DataFrame): Analytics dataframe
        region_id (str): Region ID to filter by
    
    Returns:
        dict: Region metrics
    """
    if df is None or df.empty:
        return {}
    
    # Filter by region if specified
    if region_id:
        region_data = df[df['REGIONID'] == region_id]
    else:
        region_data = df
    
    if region_data.empty:
        return {}
    
    metrics = {
        'total_records': len(region_data),
        'date_range': f"{region_data['DATE'].min()} to {region_data['DATE'].max()}"
    }
    
    # Load metrics
    if 'TOTALDEMAND' in region_data.columns:
        metrics.update({
            'avg_load_mw': region_data['TOTALDEMAND'].mean(),
            'max_load_mw': region_data['TOTALDEMAND'].max(),
            'min_load_mw': region_data['TOTALDEMAND'].min()
        })
    
    # Price metrics  
    if 'RRP' in region_data.columns:
        metrics.update({
            'avg_price': region_data['RRP'].mean(),
            'max_price': region_data['RRP'].max(),
            'min_price': region_data['RRP'].min()
        })
    
    return metrics

def calculate_forecasting_metrics(actual, predicted):
    """
    Calculate forecasting accuracy metrics
    
    Args:
        actual (pd.Series): Actual values
        predicted (pd.Series): Predicted values
    
    Returns:
        dict: Forecasting metrics
    """
    if len(actual) == 0 or len(predicted) == 0:
        return {}
    
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {}
    
    # Calculate metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'accuracy': max(0, 100 - mape)
    }

def calculate_weather_impact_metrics(df):
    """
    Calculate weather impact metrics
    
    Args:
        df (pd.DataFrame): Analytics dataframe with weather data
    
    Returns:
        dict: Weather impact metrics
    """
    if df is None or df.empty:
        return {}
    
    metrics = {}
    
    # Temperature correlation with demand
    if 'AIR_TEMP' in df.columns and 'TOTALDEMAND' in df.columns:
        temp_corr = df['AIR_TEMP'].corr(df['TOTALDEMAND'])
        metrics['temperature_demand_correlation'] = temp_corr
    
    # Weather variability
    if 'AIR_TEMP' in df.columns:
        metrics['temperature_variability'] = df['AIR_TEMP'].std()
    
    if 'HUMIDITY' in df.columns:
        metrics['humidity_variability'] = df['HUMIDITY'].std()
    
    return metrics

def calculate_savings_metrics(df, baseline_price=None):
    """
    Calculate potential savings metrics
    
    Args:
        df (pd.DataFrame): Analytics dataframe
        baseline_price (float): Baseline price for comparison
    
    Returns:
        dict: Savings metrics
    """
    if df is None or df.empty or 'RRP' not in df.columns:
        return {}
    
    if baseline_price is None:
        baseline_price = df['RRP'].mean()
    
    # Calculate potential savings
    current_avg_price = df['RRP'].mean()
    savings_per_mwh = max(0, baseline_price - current_avg_price)
    
    if 'TOTALDEMAND' in df.columns:
        total_demand = df['TOTALDEMAND'].sum()
        total_savings = savings_per_mwh * total_demand
    else:
        total_savings = 0
    
    return {
        'baseline_price': baseline_price,
        'current_avg_price': current_avg_price,
        'savings_per_mwh': savings_per_mwh,
        'total_potential_savings': total_savings,
        'savings_percentage': (savings_per_mwh / baseline_price * 100) if baseline_price > 0 else 0
    }

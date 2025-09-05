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
    
    # Customer data
    customer_data = df[df['CUSTOMER_NAME'].notna()]
    
    # Weather data
    weather_data = df[df['AIR_TEMP_mean'].notna()]
    
    # Trading data
    trading_data = df[df['RRP_mean'].notna()]
    
    metrics = {
        'total_customers': customer_data['CUSTOMER_NAME'].nunique() if not customer_data.empty else 0,
        'total_records': len(df),
        'customer_records': len(customer_data),
        'weather_records': len(weather_data),
        'trading_records': len(trading_data)
    }
    
    # Load metrics
    if not customer_data.empty:
        # Use NET_CONSUMPTION_MW or mean of available load columns
        if 'NET_CONSUMPTION_MW' in customer_data.columns:
            daily_load = customer_data.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
        elif 'HALFHOURLY_TOTAL_MW_B1' in customer_data.columns:
            # Mean B1 and E1 if available
            load_cols = ['HALFHOURLY_TOTAL_MW_B1', 'HALFHOURLY_TOTAL_MW_E1']
            available_load_cols = [col for col in load_cols if col in customer_data.columns]
            if available_load_cols:
                daily_load = customer_data.groupby('DATE')[available_load_cols].mean().sum(axis=1)
            else:
                daily_load = pd.Series()
        else:
            daily_load = pd.Series()
            
        if not daily_load.empty:
            metrics.update({
                'avg_daily_load': daily_load.mean(),
                'max_daily_load': daily_load.max(),
                'min_daily_load': daily_load.min(),
                'load_std': daily_load.std(),
                'total_days': daily_load.count()
            })
    
    # Data quality metrics
    total_possible_records = len(df)
    complete_records = len(df.dropna())
    
    metrics.update({
        'data_coverage': (len(customer_data) / total_possible_records * 100) if total_possible_records > 0 else 0,
        'data_quality_score': (complete_records / total_possible_records * 100) if total_possible_records > 0 else 0,
        'weather_coverage': (len(weather_data) / total_possible_records * 100) if total_possible_records > 0 else 0
    })
    
    return metrics

def calculate_customer_metrics(customer_data):
    """
    Calculate customer-specific metrics
    
    Args:
        customer_data (pd.DataFrame): Customer data
    
    Returns:
        dict: Customer metrics
    """
    if customer_data is None or customer_data.empty:
        return {}
    
    # Daily aggregation - use appropriate load column
    if 'NET_CONSUMPTION_MW' in customer_data.columns:
        daily_load = customer_data.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
    elif 'HALFHOURLY_TOTAL_MW_B1' in customer_data.columns:
        load_cols = ['HALFHOURLY_TOTAL_MW_B1', 'HALFHOURLY_TOTAL_MW_E1']
        available_load_cols = [col for col in load_cols if col in customer_data.columns]
        if available_load_cols:
            daily_load = customer_data.groupby('DATE')[available_load_cols].mean().sum(axis=1)
        else:
            daily_load = pd.Series()
    else:
        daily_load = pd.Series()
    
    if not daily_load.empty:
        metrics = {
            'avg_daily_consumption': daily_load.mean(),
            'peak_daily_consumption': daily_load.max(),
            'min_daily_consumption': daily_load.min(),
            'consumption_volatility': daily_load.std(),
            'total_consumption': daily_load.sum(),
            'data_days': daily_load.count(),
            'load_factor': daily_load.mean() / daily_load.max() if daily_load.max() > 0 else 0
        }
    else:
        metrics = {}
    
    # Meter type breakdown - use appropriate columns
    if 'HALFHOURLY_TOTAL_MW_B1' in customer_data.columns:
        b1_data = customer_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
        if 'HALFHOURLY_TOTAL_MW_E1' in customer_data.columns:
            e1_data = customer_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean()
            metrics['meter_breakdown'] = {
                'B1': {'mean': b1_data.mean(), 'max': b1_data.max(), 'count': len(b1_data)},
                'E1': {'mean': e1_data.mean(), 'max': e1_data.max(), 'count': len(e1_data)}
            }
    
    # Seasonal patterns
    if not daily_load.empty:
        customer_data_copy = customer_data.copy()
        customer_data_copy['MONTH'] = customer_data_copy['DATE'].dt.month
        if 'NET_CONSUMPTION_MW' in customer_data.columns:
            monthly_avg = customer_data_copy.groupby('MONTH')['NET_CONSUMPTION_MW'].mean()
        else:
            monthly_avg = pd.Series()
        if not monthly_avg.empty:
            metrics['seasonal_pattern'] = monthly_avg.to_dict()
    
    return metrics

def calculate_weather_correlation(df):
    """
    Calculate correlation between load and weather variables
    
    Args:
        df (pd.DataFrame): Combined data with load and weather
    
    Returns:
        dict: Correlation metrics
    """
    if df is None or df.empty:
        return {}
    
    # Filter to complete records
    complete_data = df[
        df['CUSTOMER_NAME'].notna() & 
        df['AIR_TEMP_mean'].notna() & 
        df['NET_CONSUMPTION_MW'].notna()
    ]
    
    if complete_data.empty:
        return {}
    
    # Daily aggregation
    daily_data = complete_data.groupby('DATE').agg({
        'NET_CONSUMPTION_MW': 'sum',
        'AIR_TEMP_mean': 'first',
        'APPARENT_TEMP_mean': 'first',
        'HUMIDITY_mean': 'first',
        'WIND_SPEED_mean': 'first'
    })
    
    # Calculate correlations
    correlations = {}
    weather_vars = ['AIR_TEMP_mean', 'APPARENT_TEMP_mean', 'HUMIDITY_mean', 'WIND_SPEED_mean']
    
    for var in weather_vars:
        if var in daily_data.columns:
            corr = daily_data['NET_CONSUMPTION_MW'].corr(daily_data[var])
            if not np.isnan(corr):
                correlations[var] = corr
    
    return correlations

def calculate_price_correlation(df):
    """
    Calculate correlation between load and electricity prices
    
    Args:
        df (pd.DataFrame): Combined data with load and prices
    
    Returns:
        dict: Price correlation metrics
    """
    if df is None or df.empty:
        return {}
    
    # Filter to complete records
    complete_data = df[
        df['CUSTOMER_NAME'].notna() & 
        df['RRP_mean'].notna() & 
        df['NET_CONSUMPTION_MW'].notna()
    ]
    
    if complete_data.empty:
        return {}
    
    # Daily aggregation
    daily_data = complete_data.groupby('DATE').agg({
        'NET_CONSUMPTION_MW': 'sum',
        'RRP_mean': 'first',
        'RRP_max': 'first',
        'RRP_min': 'first'
    })
    
    # Calculate correlations
    correlations = {}
    price_vars = ['RRP_mean', 'RRP_max', 'RRP_min']
    
    for var in price_vars:
        if var in daily_data.columns:
            corr = daily_data['NET_CONSUMPTION_MW'].corr(daily_data[var])
            if not np.isnan(corr):
                correlations[var] = corr
    
    return correlations

def calculate_load_profile_characteristics(customer_data):
    """
    Calculate load profile characteristics for pricing analysis
    
    Args:
        customer_data (pd.DataFrame): Customer meter data
    
    Returns:
        dict: Load profile characteristics
    """
    if customer_data is None or customer_data.empty:
        return {}
    
    # Use appropriate load column
    if 'NET_CONSUMPTION_MW' in customer_data.columns:
        daily_load = customer_data.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
    elif 'HALFHOURLY_TOTAL_MW_B1' in customer_data.columns:
        load_cols = ['HALFHOURLY_TOTAL_MW_B1', 'HALFHOURLY_TOTAL_MW_E1']
        available_load_cols = [col for col in load_cols if col in customer_data.columns]
        if available_load_cols:
            daily_load = customer_data.groupby('DATE')[available_load_cols].mean().sum(axis=1)
        else:
            daily_load = pd.Series()
    else:
        daily_load = pd.Series()
    
    if daily_load.empty:
        return {}
    
    # Basic statistics
    characteristics = {
        'average_load': daily_load.mean(),
        'peak_load': daily_load.max(),
        'base_load': daily_load.min(),
        'load_factor': daily_load.mean() / daily_load.max() if daily_load.max() > 0 else 0,
        'volatility': daily_load.std() / daily_load.mean() if daily_load.mean() > 0 else 0
    }
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        characteristics[f'p{p}'] = daily_load.quantile(p/100)
    
    # Peak to average ratio
    characteristics['peak_to_average_ratio'] = daily_load.max() / daily_load.mean() if daily_load.mean() > 0 else 0
    
    # Seasonal analysis
    if not daily_load.empty:
        # Create a dataframe with dates and daily loads for seasonal analysis
        seasonal_df = pd.DataFrame({
            'DATE': daily_load.index,
            'DAILY_LOAD': daily_load.values
        })
        
        # Ensure DATE is datetime type
        if not pd.api.types.is_datetime64_any_dtype(seasonal_df['DATE']):
            seasonal_df['DATE'] = pd.to_datetime(seasonal_df['DATE'])
        
        seasonal_df['MONTH'] = seasonal_df['DATE'].dt.month
        seasonal_df['SEASON'] = seasonal_df['MONTH'].map({
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        })
        
        seasonal_load = seasonal_df.groupby('SEASON')['DAILY_LOAD'].mean()
        characteristics['seasonal_variation'] = seasonal_load.to_dict()
    else:
        characteristics['seasonal_variation'] = {}
    
    return characteristics

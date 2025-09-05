"""
Data Loader Component

Handles loading and caching of analytics data for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_analytics_data():
    """
    Load the unified analytics dataset with caching
    
    Returns:
        pd.DataFrame: Unified analytics dataset
    """
    try:
        # Path to the processed data
        data_path = Path(__file__).parent.parent.parent / "processed" / "master_dataset.parquet"
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None
        
        df = pd.read_parquet(data_path)
        
        # Reset index to make SETTLEMENTDATE a column and rename to DATE
        df = df.reset_index()
        if 'SETTLEMENTDATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        else:
            df['DATE'] = pd.to_datetime(df.index)
        
        logger.info(f"Loaded analytics data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading analytics data: {e}")
        return None

@st.cache_data
def load_customer_summary():
    """
    Load customer summary data - derived from master dataset
    
    Returns:
        pd.DataFrame: Customer summary dataset
    """
    try:
        # Use main analytics data and aggregate for customer summary
        df = load_analytics_data()
        if df is None:
            return None
        
        # Create customer summary by aggregating by region and date
        customer_summary = df.groupby(['DATE', 'REGIONID']).agg({
            'TOTALDEMAND': 'mean',
            'RRP': 'mean',
            'AVAILABLEGENERATION': 'mean',
            'NETINTERCHANGE': 'mean'
        }).reset_index()
        
        return customer_summary
        
    except Exception as e:
        logger.error(f"Error loading customer summary: {e}")
        return None

@st.cache_data
def load_training_data():
    """
    Load training/weather data
    
    Returns:
        pd.DataFrame: Training data with weather variables
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "input" / "TRAINING_INDEPENDENT_INPUT.csv"
        
        if not data_path.exists():
            logger.warning(f"Training data file not found: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        df['DATE'] = pd.to_datetime(df['DATE_TIME_HH'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None

@st.cache_data
def load_trading_data():
    """
    Load electricity trading price data
    
    Returns:
        pd.DataFrame: Trading price data
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "input" / "TRADINGPRICE.csv"
        
        if not data_path.exists():
            logger.warning(f"Trading data file not found: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        df['DATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading trading data: {e}")
        return None

def get_data_quality_summary():
    """
    Get data quality summary from the analytics data
    
    Returns:
        list: List of data quality metrics
    """
    try:
        df = load_analytics_data()
        if df is None:
            return None
        
        # Generate basic data quality metrics
        quality_metrics = []
        
        # Data completeness
        total_records = len(df)
        complete_records = df.dropna().shape[0]
        completeness = (complete_records / total_records) * 100
        
        quality_metrics.append({
            'metric': 'Data Completeness',
            'value': f'{completeness:.1f}%',
            'status': 'Good' if completeness > 95 else 'Needs Attention'
        })
        
        # Date range
        date_range = f"{df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}"
        quality_metrics.append({
            'metric': 'Date Range',
            'value': date_range,
            'status': 'Good'
        })
        
        # Region coverage
        regions = df['REGIONID'].nunique()
        quality_metrics.append({
            'metric': 'Regions Covered',
            'value': f'{regions} regions',
            'status': 'Good'
        })
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error generating data quality summary: {e}")
        return None

def get_data_date_range():
    """
    Get the date range of available data
    
    Returns:
        tuple: (start_date, end_date)
    """
    df = load_analytics_data()
    if df is not None:
        return df['DATE'].min(), df['DATE'].max()
    return None, None

def filter_data_by_date_range(df, start_date, end_date):
    """
    Filter dataframe by date range
    
    Args:
        df (pd.DataFrame): Input dataframe
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None:
        return None
    
    return df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]

def get_customer_list():
    """
    Get list of available regions (customers)
    
    Returns:
        list: List of region IDs
    """
    df = load_analytics_data()
    if df is not None:
        regions = df['REGIONID'].unique().tolist()
        return sorted(regions)
    return []

def filter_customer_data(region_id=None):
    """
    Filter data for specific region
    
    Args:
        region_id (str): Region ID to filter by
    
    Returns:
        pd.DataFrame: Filtered region data
    """
    df = load_analytics_data()
    if df is None:
        return None
    
    if region_id:
        return df[df['REGIONID'] == region_id]
    else:
        return df

@st.cache_data
def load_meter_data_raw():
    """
    Load raw meter data for detailed analysis
    Uses the master dataset as meter data
    
    Returns:
        pd.DataFrame: Raw meter data
    """
    try:
        # Use the main analytics data as meter data
        return load_analytics_data()
        
    except Exception as e:
        logger.error(f"Error loading meter data: {e}")
        return None

def validate_data_availability():
    """
    Validate that required data is available
    
    Returns:
        dict: Data availability status
    """
    status = {
        'analytics_data': load_analytics_data() is not None,
        'customer_summary': load_customer_summary() is not None,
        'training_data': load_training_data() is not None,
        'trading_data': load_trading_data() is not None,
        'data_quality': get_data_quality_summary() is not None
    }
    
    return status

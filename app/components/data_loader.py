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
        # Path to the output data
        data_path = Path(__file__).parent.parent.parent / "output" / "unified_analytics.parquet"
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None
        
        df = pd.read_parquet(data_path)
        
        # Ensure DATE column is datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        logger.info(f"Loaded analytics data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading analytics data: {e}")
        return None

@st.cache_data
def load_customer_summary():
    """
    Load customer summary data
    
    Returns:
        pd.DataFrame: Customer summary dataset
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "output" / "customer_summary.parquet"
        
        if not data_path.exists():
            return None
        
        df = pd.read_parquet(data_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        return df
        
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
        data_path = Path(__file__).parent.parent.parent / "output" / "training_summary.parquet"
        
        if not data_path.exists():
            return None
        
        df = pd.read_parquet(data_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
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
        data_path = Path(__file__).parent.parent.parent / "output" / "trading_summary.parquet"
        
        if not data_path.exists():
            return None
        
        df = pd.read_parquet(data_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading trading data: {e}")
        return None

def get_data_quality_summary():
    """
    Get data quality summary from the quality report
    
    Returns:
        list: List of data quality metrics
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "output" / "data_quality_report.csv"
        
        if not data_path.exists():
            return None
        
        df = pd.read_csv(data_path)
        
        # Convert to list of dictionaries for display
        return df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error loading data quality report: {e}")
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
    Get list of available customers
    
    Returns:
        list: List of customer names
    """
    df = load_analytics_data()
    if df is not None:
        customers = df[df['CUSTOMER_NAME'].notna()]['CUSTOMER_NAME'].unique().tolist()
        return customers
    return []

def filter_customer_data(customer_name=None):
    """
    Filter data for specific customer
    
    Args:
        customer_name (str): Customer name to filter by
    
    Returns:
        pd.DataFrame: Filtered customer data
    """
    df = load_analytics_data()
    if df is None:
        return None
    
    if customer_name:
        return df[df['CUSTOMER_NAME'] == customer_name]
    else:
        return df[df['CUSTOMER_NAME'].notna()]

@st.cache_data
def load_meter_data_raw():
    """
    Load raw meter data for detailed analysis
    
    Returns:
        pd.DataFrame: Raw meter data
    """
    try:
        data_path = Path(__file__).parent.parent.parent / "output" / "meter_data.parquet"
        
        if not data_path.exists():
            return None
        
        df = pd.read_parquet(data_path)
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading raw meter data: {e}")
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

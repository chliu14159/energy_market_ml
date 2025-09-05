"""
Utility functions for Retail Analytics Data Preparation

This module contains helper functions used across the data preparation pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def profile_dataframe(df, name="DataFrame"):
    """
    Generate a comprehensive profile of a DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame to profile
        name (str): Name for the DataFrame (for logging)
    
    Returns:
        dict: Profile information
    """
    profile = {
        'name': name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Date range if DATE column exists
    if 'DATE' in df.columns:
        profile['date_range'] = {
            'min': df['DATE'].min(),
            'max': df['DATE'].max(),
            'count': df['DATE'].nunique()
        }
    
    return profile

def print_dataframe_profile(profile):
    """Pretty print DataFrame profile"""
    print(f"\n📊 {profile['name'].upper()}")
    print("-" * 40)
    print(f"Shape: {profile['shape']}")
    print(f"Memory: {profile['memory_mb']:.1f} MB")
    print(f"Columns: {len(profile['columns'])}")
    
    if 'date_range' in profile:
        print(f"Date range: {profile['date_range']['min']} to {profile['date_range']['max']}")
    
    # Show columns with missing data
    missing_cols = {k: v for k, v in profile['missing_values'].items() if v > 0}
    if missing_cols:
        print("Missing data:")
        for col, count in missing_cols.items():
            pct = profile['missing_percentage'][col]
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print("✅ No missing data")

def detect_date_columns(df):
    """
    Automatically detect date columns in a DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
    
    Returns:
        list: List of potential date column names
    """
    date_keywords = ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']
    date_cols = []
    
    for col in df.columns:
        if any(keyword in col.upper() for keyword in date_keywords):
            date_cols.append(col)
    
    return date_cols

def standardize_date_column(df, source_col, target_col='DATE'):
    """
    Standardize a date column to datetime and create a date-only column
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        source_col (str): Source date column name
        target_col (str): Target column name for date
    
    Returns:
        pandas.DataFrame: DataFrame with standardized date columns
    """
    df = df.copy()
    
    if source_col in df.columns:
        # Convert to datetime
        df[source_col] = pd.to_datetime(df[source_col])
        
        # Create date-only column
        df[target_col] = df[source_col].dt.date
        df[target_col] = pd.to_datetime(df[target_col])
        
        logger.info(f"Standardized date column: {source_col} -> {target_col}")
    else:
        logger.warning(f"Source column {source_col} not found in DataFrame")
    
    return df

def aggregate_to_daily(df, group_cols, agg_dict, date_col='DATE'):
    """
    Aggregate data to daily level
    
    Args:
        df (pandas.DataFrame): DataFrame to aggregate
        group_cols (list): Columns to group by (should include date column)
        agg_dict (dict): Aggregation specification
        date_col (str): Date column name
    
    Returns:
        pandas.DataFrame: Aggregated DataFrame
    """
    if date_col not in group_cols:
        group_cols = [date_col] + group_cols
    
    # Perform aggregation
    agg_df = df.groupby(group_cols).agg(agg_dict).round(4)
    
    # Flatten column names if multi-level
    if isinstance(agg_df.columns, pd.MultiIndex):
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
    
    agg_df = agg_df.reset_index()
    
    logger.info(f"Aggregated to daily level: {df.shape} -> {agg_df.shape}")
    
    return agg_df

def save_dataframe(df, filepath, formats=['csv', 'parquet']):
    """
    Save DataFrame in multiple formats
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        filepath (str or Path): Base filepath (without extension)
        formats (list): List of formats to save ['csv', 'parquet']
    """
    filepath = Path(filepath)
    
    for fmt in formats:
        if fmt == 'csv':
            output_path = filepath.with_suffix('.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV: {output_path}")
        elif fmt == 'parquet':
            output_path = filepath.with_suffix('.parquet')
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved Parquet: {output_path}")
        else:
            logger.warning(f"Unsupported format: {fmt}")

def calculate_data_quality_metrics(df):
    """
    Calculate data quality metrics for a DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
    
    Returns:
        dict: Data quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_cells': len(df) * len(df.columns),
        'missing_cells': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'completeness_pct': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    return metrics

def create_data_quality_report(datasets):
    """
    Create a comprehensive data quality report for multiple datasets
    
    Args:
        datasets (dict): Dictionary of dataset_name -> DataFrame
    
    Returns:
        pandas.DataFrame: Data quality report
    """
    quality_data = []
    
    for name, df in datasets.items():
        metrics = calculate_data_quality_metrics(df)
        metrics['dataset_name'] = name
        quality_data.append(metrics)
    
    quality_df = pd.DataFrame(quality_data)
    quality_df = quality_df[['dataset_name'] + [col for col in quality_df.columns if col != 'dataset_name']]
    
    return quality_df

def filter_by_region(df, region_col='REGIONID', target_region='QLD1'):
    """
    Filter DataFrame by region
    
    Args:
        df (pandas.DataFrame): DataFrame to filter
        region_col (str): Region column name
        target_region (str): Target region to keep
    
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    if region_col in df.columns:
        filtered_df = df[df[region_col] == target_region].copy()
        logger.info(f"Filtered by region {target_region}: {df.shape} -> {filtered_df.shape}")
        return filtered_df
    else:
        logger.warning(f"Region column {region_col} not found")
        return df

def merge_with_validation(left_df, right_df, on=None, how='left', validate=None):
    """
    Merge DataFrames with validation and logging
    
    Args:
        left_df (pandas.DataFrame): Left DataFrame
        right_df (pandas.DataFrame): Right DataFrame
        on (str or list): Column(s) to merge on
        how (str): Type of merge
        validate (str): Validation type for merge
    
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    initial_shape = left_df.shape
    
    # Handle pandas merge parameters properly
    merge_kwargs = {'left': left_df, 'right': right_df, 'on': on, 'how': how}
    if validate is not None:
        merge_kwargs['validate'] = validate
    
    merged_df = pd.merge(**merge_kwargs)
    
    logger.info(f"Merge completed: {initial_shape} -> {merged_df.shape}")
    
    return merged_df

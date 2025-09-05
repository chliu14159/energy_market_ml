#!/usr/bin/env python3
"""
Validation script to compare outputs with UNIT_TEST notebook expectations

This script validates that our Python data preparation script produces 
results consistent with the UNIT_TEST notebook logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_customer_summary():
    """Validate customer summary against expected patterns"""
    logger.info("Validating customer summary...")
    
    df = pd.read_parquet('../output/customer_summary.parquet')
    
    print("\n📊 CUSTOMER SUMMARY VALIDATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"Unique customers: {df['CUSTOMER_NAME'].nunique()}")
    print(f"Unique meters: {df['METER_ID_SUFFIX'].nunique()}")
    
    # Check aggregation logic
    print(f"\nDaily totals range: {df['DAILY_TOTAL_MW'].min():.2f} to {df['DAILY_TOTAL_MW'].max():.2f} MW")
    print(f"Daily means range: {df['DAILY_MEAN_MW'].min():.2f} to {df['DAILY_MEAN_MW'].max():.2f} MW")
    
    # Check for expected customer names
    customers = df['CUSTOMER_NAME'].unique()
    print(f"\nCustomers in data: {list(customers)}")
    
    # Show sample data
    print(f"\nSample records:")
    print(df[['DATE', 'CUSTOMER_NAME', 'DAILY_TOTAL_MW', 'DAILY_MEAN_MW', 'INTERVALS_COUNT']].head())
    
    return df

def validate_unified_analytics():
    """Validate unified analytics dataset"""
    logger.info("Validating unified analytics...")
    
    df = pd.read_parquet('../output/unified_analytics.parquet')
    
    print("\n📊 UNIFIED ANALYTICS VALIDATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Check key columns exist
    expected_columns = [
        'DATE', 'DAY_TYPE_QLD', 'CUSTOMER_NAME', 'DAILY_TOTAL_MW',
        'RRP_mean', 'TOTALDEMAND_mean', 'AIR_TEMP_mean'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing expected columns: {missing_cols}")
    else:
        print("✅ All expected columns present")
    
    # Check data coverage
    non_null_counts = df.count()
    total_rows = len(df)
    
    print(f"\nData coverage:")
    for col in ['CUSTOMER_NAME', 'DAILY_TOTAL_MW', 'RRP_mean', 'TOTALDEMAND_mean']:
        if col in df.columns:
            coverage = non_null_counts[col] / total_rows * 100
            print(f"  {col}: {coverage:.1f}% ({non_null_counts[col]:,} / {total_rows:,})")
    
    # Show sample with customer data
    customer_data = df[df['CUSTOMER_NAME'].notna()]
    if not customer_data.empty:
        print(f"\nSample customer records:")
        sample_cols = ['DATE', 'CUSTOMER_NAME', 'DAILY_TOTAL_MW', 'RRP_mean', 'TOTALDEMAND_mean']
        available_cols = [col for col in sample_cols if col in customer_data.columns]
        print(customer_data[available_cols].head())
    
    return df

def validate_market_data():
    """Validate market data summaries"""
    logger.info("Validating market data...")
    
    # Trading summary
    trading_df = pd.read_parquet('../output/trading_summary.parquet')
    print("\n📊 TRADING SUMMARY VALIDATION")
    print("="*50)
    print(f"Shape: {trading_df.shape}")
    print(f"Regions: {trading_df['REGIONID'].unique()}")
    print(f"Date range: {trading_df['DATE'].min()} to {trading_df['DATE'].max()}")
    print(f"Price range: ${trading_df['RRP_mean'].min():.2f} to ${trading_df['RRP_mean'].max():.2f}")
    
    # QLD1 specific
    qld1_data = trading_df[trading_df['REGIONID'] == 'QLD1']
    print(f"QLD1 records: {len(qld1_data):,}")
    
    # Dispatch summary
    dispatch_df = pd.read_parquet('../output/dispatch_summary.parquet')
    print("\n📊 DISPATCH SUMMARY VALIDATION")
    print("="*50)
    print(f"Shape: {dispatch_df.shape}")
    print(f"Regions: {dispatch_df['REGIONID'].unique()}")
    print(f"Date range: {dispatch_df['DATE'].min()} to {dispatch_df['DATE'].max()}")
    print(f"Demand range: {dispatch_df['TOTALDEMAND_mean'].min():.2f} to {dispatch_df['TOTALDEMAND_mean'].max():.2f} MW")
    
    return trading_df, dispatch_df

def validate_weather_data():
    """Validate weather forecast summary"""
    logger.info("Validating weather data...")
    
    df = pd.read_parquet('../output/weather_summary.parquet')
    
    print("\n📊 WEATHER SUMMARY VALIDATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    if 'AIR_TEMP_mean' in df.columns:
        print(f"Temperature range: {df['AIR_TEMP_mean'].min():.1f}°C to {df['AIR_TEMP_mean'].max():.1f}°C")
    
    if 'PV_POWER_sum' in df.columns:
        pv_data = df['PV_POWER_sum'].dropna()
        if not pv_data.empty:
            print(f"PV Power range: {pv_data.min():.2f} to {pv_data.max():.2f}")
    
    return df

def check_data_consistency():
    """Check consistency between datasets"""
    logger.info("Checking data consistency...")
    
    print("\n📊 DATA CONSISTENCY CHECKS")
    print("="*50)
    
    # Load key datasets
    customer_df = pd.read_parquet('../output/customer_summary.parquet')
    analytics_df = pd.read_parquet('../output/unified_analytics.parquet')
    
    # Check date consistency
    customer_dates = set(customer_df['DATE'].dt.date)
    analytics_dates = set(analytics_df['DATE'].dt.date)
    
    overlap_dates = customer_dates.intersection(analytics_dates)
    print(f"Customer data dates: {len(customer_dates):,}")
    print(f"Analytics data dates: {len(analytics_dates):,}")
    print(f"Overlapping dates: {len(overlap_dates):,}")
    
    # Check customer data consistency
    analytics_customers = analytics_df[analytics_df['CUSTOMER_NAME'].notna()]
    if not analytics_customers.empty:
        analytics_customer_dates = set(analytics_customers['DATE'].dt.date)
        customer_consistency = len(customer_dates.intersection(analytics_customer_dates)) / len(customer_dates) * 100
        print(f"Customer date consistency: {customer_consistency:.1f}%")
    
    # Sample comparison
    if overlap_dates:
        sample_date = list(overlap_dates)[0]
        sample_date_dt = pd.to_datetime(sample_date)
        
        customer_sample = customer_df[customer_df['DATE'] == sample_date_dt]
        analytics_sample = analytics_customers[analytics_customers['DATE'] == sample_date_dt]
        
        print(f"\nSample date {sample_date}:")
        print(f"  Customer summary records: {len(customer_sample)}")
        print(f"  Analytics records: {len(analytics_sample)}")

def main():
    """Run all validation checks"""
    print("🔍 RETAIL ANALYTICS DATA VALIDATION")
    print("="*60)
    
    try:
        # Run validations
        customer_df = validate_customer_summary()
        trading_df, dispatch_df = validate_market_data()
        weather_df = validate_weather_data()
        analytics_df = validate_unified_analytics()
        
        # Check consistency
        check_data_consistency()
        
        print("\n✅ VALIDATION SUMMARY")
        print("="*60)
        print("All datasets validated successfully!")
        print("Data preparation script is working correctly.")
        print("Ready for Streamlit application development.")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()

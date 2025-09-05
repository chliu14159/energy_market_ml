#!/usr/bin/env python3
"""
Customer Load Forecasting - Data Analysis

This script analyzes the processed data to understand UNIT_TEST_Elanor's 
load patterns and relationships with weather/market variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_analytics_data():
    """Load the unified analytics dataset"""
    return pd.read_parquet('../output/unified_analytics.parquet')

def analyze_customer_load_patterns():
    """Analyze UNIT_TEST_Elanor load patterns"""
    print("📊 UNIT_TEST_ELANOR LOAD PATTERN ANALYSIS")
    print("="*60)
    
    df = load_analytics_data()
    
    # Filter to customer data only
    customer_data = df[df['CUSTOMER_NAME'].notna()].copy()
    
    print(f"Customer data records: {len(customer_data):,}")
    print(f"Date range: {customer_data['DATE'].min()} to {customer_data['DATE'].max()}")
    print(f"Total days: {customer_data['DATE'].nunique():,}")
    
    # Meter types analysis
    meter_summary = customer_data.groupby('METER_ID_SUFFIX').agg({
        'DAILY_TOTAL_MW': ['count', 'mean', 'max', 'min', 'std'],
        'DATE': ['min', 'max']
    }).round(3)
    
    print(f"\n📍 Meter Types:")
    print(meter_summary)
    
    # Daily statistics
    daily_stats = customer_data.groupby('DATE').agg({
        'DAILY_TOTAL_MW': 'sum',  # Total across both meters
        'AIR_TEMP_mean': 'first',
        'RRP_mean': 'first'
    }).dropna()
    
    print(f"\n📈 Daily Load Statistics:")
    print(f"Average daily consumption: {daily_stats['DAILY_TOTAL_MW'].mean():.2f} MW")
    print(f"Maximum daily consumption: {daily_stats['DAILY_TOTAL_MW'].max():.2f} MW")
    print(f"Minimum daily consumption: {daily_stats['DAILY_TOTAL_MW'].min():.2f} MW")
    print(f"Standard deviation: {daily_stats['DAILY_TOTAL_MW'].std():.2f} MW")
    
    return customer_data, daily_stats

def analyze_weather_relationships():
    """Analyze relationships between load and weather variables"""
    print("\n🌡️ WEATHER-LOAD RELATIONSHIP ANALYSIS")
    print("="*60)
    
    df = load_analytics_data()
    
    # Get complete records with customer, weather, and price data
    complete_data = df[
        df['CUSTOMER_NAME'].notna() & 
        df['AIR_TEMP_mean'].notna() & 
        df['RRP_mean'].notna()
    ].copy()
    
    print(f"Complete records for analysis: {len(complete_data):,}")
    
    if len(complete_data) > 0:
        # Aggregate by date (sum across meter types)
        daily_analysis = complete_data.groupby('DATE').agg({
            'DAILY_TOTAL_MW': 'sum',
            'AIR_TEMP_mean': 'first',
            'APPARENT_TEMP_mean': 'first',
            'HUMIDITY_mean': 'first',
            'WIND_SPEED_mean': 'first',
            'RRP_mean': 'first',
            'IS_WORKDAY_first': 'first',
            'IS_SCHOOL_HOLIDAY_first': 'first'
        }).reset_index()
        
        # Calculate correlations
        correlations = daily_analysis[
            ['DAILY_TOTAL_MW', 'AIR_TEMP_mean', 'APPARENT_TEMP_mean', 
             'HUMIDITY_mean', 'WIND_SPEED_mean', 'RRP_mean']
        ].corr()['DAILY_TOTAL_MW'].sort_values(ascending=False)
        
        print(f"\n📊 Load Correlations with External Variables:")
        for var, corr in correlations.items():
            if var != 'DAILY_TOTAL_MW':
                print(f"   {var}: {corr:+.3f}")
        
        # Seasonal patterns
        daily_analysis['MONTH'] = daily_analysis['DATE'].dt.month
        monthly_avg = daily_analysis.groupby('MONTH')['DAILY_TOTAL_MW'].mean()
        
        print(f"\n📅 Monthly Load Averages:")
        for month, avg_load in monthly_avg.items():
            month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
            print(f"   {month_name}: {avg_load:.2f} MW")
        
        return daily_analysis
    
    return None

def analyze_market_context():
    """Analyze electricity market context"""
    print("\n💰 ELECTRICITY MARKET CONTEXT")
    print("="*60)
    
    df = load_analytics_data()
    
    # Market price analysis
    price_data = df[df['RRP_mean'].notna()].copy()
    
    print(f"Market price records: {len(price_data):,}")
    print(f"Price data range: {price_data['DATE'].min()} to {price_data['DATE'].max()}")
    
    print(f"\n💵 QLD1 Price Statistics:")
    print(f"Average price: ${price_data['RRP_mean'].mean():.2f}/MWh")
    print(f"Maximum price: ${price_data['RRP_max'].max():.2f}/MWh")
    print(f"Minimum price: ${price_data['RRP_min'].min():.2f}/MWh")
    print(f"Price volatility (std): ${price_data['RRP_std'].mean():.2f}")
    
    # High/low price days
    high_price_threshold = price_data['RRP_mean'].quantile(0.9)
    low_price_threshold = price_data['RRP_mean'].quantile(0.1)
    
    print(f"\n📊 Price Extremes:")
    print(f"High price threshold (90th percentile): ${high_price_threshold:.2f}/MWh")
    print(f"Low price threshold (10th percentile): ${low_price_threshold:.2f}/MWh")
    
    return price_data

def create_forecast_readiness_report():
    """Generate a readiness report for load forecasting"""
    print("\n🎯 LOAD FORECASTING READINESS REPORT")
    print("="*60)
    
    df = load_analytics_data()
    
    # Data availability assessment
    customer_records = len(df[df['CUSTOMER_NAME'].notna()])
    weather_records = len(df[df['AIR_TEMP_mean'].notna()])
    price_records = len(df[df['RRP_mean'].notna()])
    
    complete_records = len(df[
        df['CUSTOMER_NAME'].notna() & 
        df['AIR_TEMP_mean'].notna() & 
        df['RRP_mean'].notna()
    ])
    
    print(f"📈 Data Availability:")
    print(f"   Customer load data: {customer_records:,} records")
    print(f"   Weather data: {weather_records:,} records") 
    print(f"   Market price data: {price_records:,} records")
    print(f"   Complete records (all variables): {complete_records:,} records")
    
    # Feature summary
    customer_data = df[df['CUSTOMER_NAME'].notna()]
    if not customer_data.empty:
        date_range_days = (customer_data['DATE'].max() - customer_data['DATE'].min()).days
        
        print(f"\n🔢 Modeling Features Available:")
        print(f"   Target Variable: DAILY_TOTAL_MW (customer load)")
        print(f"   Weather Features: Temperature, Humidity, Wind Speed")
        print(f"   Market Features: Electricity prices (RRP)")
        print(f"   Calendar Features: Workday, School Holiday indicators")
        print(f"   Time Series Length: {date_range_days} days")
        
        # Meter type breakdown
        meter_counts = customer_data['METER_ID_SUFFIX'].value_counts()
        print(f"\n🔌 Meter Types:")
        for meter, count in meter_counts.items():
            meter_type = "Battery" if meter == "B1" else "Export" if meter == "E1" else "Unknown"
            print(f"   {meter} ({meter_type}): {count:,} daily records")
    
    print(f"\n✅ Ready for Load Forecasting Models:")
    print(f"   • Time series forecasting (ARIMA, LSTM)")
    print(f"   • Regression models with weather variables")
    print(f"   • Machine learning (Random Forest, XGBoost)")
    print(f"   • Peak demand prediction")
    print(f"   • Price-responsive load modeling")

def main():
    """Run complete data analysis"""
    print("🔍 CUSTOMER LOAD FORECASTING - DATA ANALYSIS")
    print("="*70)
    
    try:
        # Load pattern analysis
        customer_data, daily_stats = analyze_customer_load_patterns()
        
        # Weather relationship analysis  
        weather_analysis = analyze_weather_relationships()
        
        # Market context analysis
        price_analysis = analyze_market_context()
        
        # Forecasting readiness
        create_forecast_readiness_report()
        
        print(f"\n🎉 ANALYSIS COMPLETE")
        print("="*70)
        print("Data is ready for customer load forecasting model development!")
        print("Focus: UNIT_TEST_Elanor load patterns + weather drivers + market context")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()

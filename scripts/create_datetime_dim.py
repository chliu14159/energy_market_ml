#!/usr/bin/env python3
"""
Create DATE_TIME_DIM for Half-Hourly Energy Market Data

This script creates a comprehensive datetime dimension table that supports
half-hourly intervals (48 periods per day) as used in the NEM energy market.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_datetime_dimension(start_date='2023-01-01', end_date='2025-12-31'):
    """
    Create a comprehensive datetime dimension for half-hourly energy market data
    
    Parameters:
    - start_date: Start date for the dimension
    - end_date: End date for the dimension
    
    Returns:
    - DataFrame with datetime dimension including PERIOD_HH (1-48)
    """
    
    print(f"🕐 Creating DATE_TIME_DIM from {start_date} to {end_date}")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create half-hourly periods (1-48 for each day)
    periods_per_day = 48
    datetime_records = []
    
    for date in date_range:
        for period_hh in range(1, periods_per_day + 1):
            # Calculate time offset (period 1 = 00:30, period 2 = 01:00, etc.)
            # NEM convention: Period 1 ends at 00:30, Period 48 ends at 00:00 (next day)
            minutes_offset = period_hh * 30
            hours_offset = (minutes_offset // 60) % 24
            minutes_remainder = minutes_offset % 60
            
            # Create datetime for this period (end time of the period)
            if period_hh == 48:
                # Period 48 ends at 00:00 of next day
                datetime_hh = (date + timedelta(days=1)).replace(hour=0, minute=0)
            else:
                datetime_hh = date.replace(hour=hours_offset, minute=minutes_remainder)
            
            # Calendar features
            record = {
                # Primary keys
                'DATE': date.date(),
                'PERIOD_HH': period_hh,
                'DATE_TIME_HH': datetime_hh,
                
                # Date components
                'YEAR': date.year,
                'MONTH': date.month,
                'DAY': date.day,
                'QUARTER': date.quarter,
                'WEEK_OF_YEAR': date.isocalendar().week,
                'DAY_OF_YEAR': date.dayofyear,
                'DAY_OF_WEEK': date.dayofweek + 1,  # Monday=1, Sunday=7
                'DAY_NAME': date.strftime('%A'),
                'MONTH_NAME': date.strftime('%B'),
                
                # Time components
                'HOUR': hours_offset,
                'MINUTE': minutes_remainder,
                'TIME_OF_DAY': f"{hours_offset:02d}:{minutes_remainder:02d}",
                
                # Business day indicators
                'IS_WEEKEND': date.dayofweek >= 5,
                'IS_WEEKDAY': date.dayofweek < 5,
                'IS_MONDAY': date.dayofweek == 0,
                'IS_FRIDAY': date.dayofweek == 4,
                
                # Energy market specific periods
                'IS_PEAK_PERIOD': 14 <= period_hh <= 38,  # 7:00 AM to 7:00 PM
                'IS_OFF_PEAK_PERIOD': period_hh < 14 or period_hh > 38,
                'IS_SHOULDER_PERIOD': period_hh in [12, 13, 39, 40],  # 6:00-7:00 AM, 7:00-8:00 PM
                
                # Time of day categories
                'TIME_CATEGORY': get_time_category(period_hh),
                'DEMAND_PERIOD': get_demand_period(period_hh, date.dayofweek),
                
                # Seasonal features for ML
                'MONTH_SIN': np.sin(2 * np.pi * date.month / 12),
                'MONTH_COS': np.cos(2 * np.pi * date.month / 12),
                'DAY_OF_WEEK_SIN': np.sin(2 * np.pi * (date.dayofweek + 1) / 7),
                'DAY_OF_WEEK_COS': np.cos(2 * np.pi * (date.dayofweek + 1) / 7),
                'HOUR_SIN': np.sin(2 * np.pi * hours_offset / 24),
                'HOUR_COS': np.cos(2 * np.pi * hours_offset / 24),
                
                # Energy trading specific
                'TRADING_DAY': date.strftime('%Y-%m-%d'),
                'PERIOD_ID': f"{date.strftime('%Y%m%d')}-{period_hh:02d}",
            }
            
            datetime_records.append(record)
    
    # Create DataFrame
    datetime_dim = pd.DataFrame(datetime_records)
    
    print(f"✅ Created {len(datetime_dim):,} datetime records")
    print(f"   Date range: {datetime_dim['DATE'].min()} to {datetime_dim['DATE'].max()}")
    print(f"   Period range: {datetime_dim['PERIOD_HH'].min()} to {datetime_dim['PERIOD_HH'].max()}")
    print(f"   Datetime range: {datetime_dim['DATE_TIME_HH'].min()} to {datetime_dim['DATE_TIME_HH'].max()}")
    
    return datetime_dim

def get_time_category(period_hh):
    """
    Categorize time periods for energy market analysis
    """
    if 1 <= period_hh <= 12:  # 00:30 - 06:00
        return 'Night'
    elif 13 <= period_hh <= 24:  # 06:30 - 12:00
        return 'Morning'
    elif 25 <= period_hh <= 36:  # 12:30 - 18:00
        return 'Afternoon'
    elif 37 <= period_hh <= 48:  # 18:30 - 24:00
        return 'Evening'
    else:
        return 'Unknown'

def get_demand_period(period_hh, day_of_week):
    """
    Classify demand periods based on NEM market structure
    """
    is_weekend = day_of_week >= 5
    
    if is_weekend:
        return 'Weekend'
    elif 14 <= period_hh <= 38:  # 7:00 AM to 7:00 PM weekdays
        return 'Peak'
    elif period_hh in [12, 13, 39, 40]:  # 6:00-7:00 AM, 7:00-8:00 PM
        return 'Shoulder'
    else:
        return 'Off-Peak'

def add_holiday_information(datetime_dim):
    """
    Add Australian holiday information (simplified version)
    This should be enhanced with actual holiday data
    """
    # For now, just add placeholder columns
    datetime_dim['IS_PUBLIC_HOLIDAY'] = False
    datetime_dim['IS_SCHOOL_HOLIDAY'] = False
    datetime_dim['HOLIDAY_NAME'] = None
    
    # Add some common Australian holidays (simplified)
    for idx, row in datetime_dim.iterrows():
        date = pd.to_datetime(row['DATE'])
        
        # New Year's Day
        if date.month == 1 and date.day == 1:
            datetime_dim.at[idx, 'IS_PUBLIC_HOLIDAY'] = True
            datetime_dim.at[idx, 'HOLIDAY_NAME'] = 'New Year\'s Day'
        
        # Australia Day
        elif date.month == 1 and date.day == 26:
            datetime_dim.at[idx, 'IS_PUBLIC_HOLIDAY'] = True
            datetime_dim.at[idx, 'HOLIDAY_NAME'] = 'Australia Day'
        
        # Christmas Day
        elif date.month == 12 and date.day == 25:
            datetime_dim.at[idx, 'IS_PUBLIC_HOLIDAY'] = True
            datetime_dim.at[idx, 'HOLIDAY_NAME'] = 'Christmas Day'
        
        # Boxing Day
        elif date.month == 12 and date.day == 26:
            datetime_dim.at[idx, 'IS_PUBLIC_HOLIDAY'] = True
            datetime_dim.at[idx, 'HOLIDAY_NAME'] = 'Boxing Day'
    
    return datetime_dim

def main():
    """
    Main function to create and save DATE_TIME_DIM
    """
    
    # Set up paths
    INPUT_PATH = Path('../input')
    
    print("🏗️ Creating Half-Hourly DATE_TIME_DIM for Energy Market")
    print("=" * 60)
    
    # Create datetime dimension
    datetime_dim = create_datetime_dimension()
    
    # Add holiday information
    print("\n🎄 Adding holiday information...")
    datetime_dim = add_holiday_information(datetime_dim)
    
    # Save to CSV
    output_file = INPUT_PATH / 'DATE_TIME_DIM.csv'
    datetime_dim.to_csv(output_file, index=False)
    
    print(f"\n💾 DATE_TIME_DIM saved to {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Display sample data
    print("\n📊 Sample Data:")
    print(datetime_dim.head(10))
    
    print("\n📈 Period Distribution:")
    print(datetime_dim.groupby(['TIME_CATEGORY', 'DEMAND_PERIOD']).size())
    
    print("\n🎯 Peak vs Off-Peak Distribution:")
    print(datetime_dim['DEMAND_PERIOD'].value_counts())
    
    print("\n✅ DATE_TIME_DIM creation complete!")
    
    return datetime_dim

if __name__ == "__main__":
    datetime_dim = main()

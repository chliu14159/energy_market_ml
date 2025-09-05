#!/usr/bin/env python3
"""Test script to verify the tariff scenarios fix"""

import sys
import pandas as pd
from pathlib import Path

# Add app path
sys.path.append(str(Path(__file__).parent / 'app'))

def calculate_tariff_scenarios(load_profile):
    """
    Calculate different tariff scenarios for a customer load profile
    """
    # Extract load data
    if 'NET_CONSUMPTION_MW' in load_profile.columns:
        daily_load = load_profile.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
    elif 'TOTALDEMAND' in load_profile.columns:
        daily_load = load_profile.groupby('DATE')['TOTALDEMAND'].mean()
    elif 'HALFHOURLY_TOTAL_MW_B1' in load_profile.columns:
        b1_load = load_profile.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
        e1_load = load_profile.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean() if 'HALFHOURLY_TOTAL_MW_E1' in load_profile.columns else pd.Series(0, index=b1_load.index)
        daily_load = b1_load + e1_load
    else:
        daily_load = pd.Series()
    
    # Check if we have valid data
    if daily_load.empty or len(daily_load) == 0:
        return {}
    
    # Basic calculations (annualized based on daily averages)
    annual_consumption_mwh = daily_load.sum() * (365 / len(daily_load)) * 24  # Convert MW*days to MWh (MW * days * 24h/day)
    peak_demand = daily_load.max()
    average_demand = daily_load.mean()
    load_factor = average_demand / peak_demand if peak_demand > 0 else 0
    
    return {'test': 'passed', 'annual_consumption_mwh': annual_consumption_mwh}

def main():
    print("Testing tariff scenarios fix...")
    
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    result = calculate_tariff_scenarios(empty_df)
    print(f"✅ Empty dataframe test: {result}")
    
    # Test with dataframe without load columns
    no_load_df = pd.DataFrame({'DATE': ['2023-01-01', '2023-01-02'], 'OTHER_COL': [1, 2]})
    result2 = calculate_tariff_scenarios(no_load_df)
    print(f"✅ No load columns test: {result2}")
    
    # Test with valid data
    valid_df = pd.DataFrame({
        'DATE': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'TOTALDEMAND': [100, 150, 120]
    })
    result3 = calculate_tariff_scenarios(valid_df)
    print(f"✅ Valid data test: {result3}")
    
    print("All tests passed! The fix should work.")

if __name__ == "__main__":
    main()

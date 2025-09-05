import pandas as pd
import numpy as np

# Load and examine the master dataset
df = pd.read_parquet('processed/master_dataset.parquet')
df = df.reset_index()

print('=== DATASET ANALYSIS ===')
print('Shape:', df.shape)
print('Date range:', df['SETTLEMENTDATE'].min(), 'to', df['SETTLEMENTDATE'].max())
print()

print('=== COLUMNS AVAILABLE ===')
for i, col in enumerate(df.columns, 1):
    print(f'{i:2}. {col}')

print()
print('=== WEATHER DATA CHECK ===')
weather_cols = ['AIR_TEMP', 'HUMIDITY', 'DEW_POINT_TEMP', 'WIND_SPEED', 'APPARENT_TEMP']
for col in weather_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct_available = (non_null / len(df)) * 100
        if non_null > 0:
            sample_values = df[col].dropna().head(3).tolist()
            print(f'{col}: {non_null:,} records ({pct_available:.1f}%) - Sample: {sample_values}')
        else:
            print(f'{col}: {non_null:,} records ({pct_available:.1f}%) - ALL NULL')
    else:
        print(f'{col}: NOT FOUND')

print()
print('=== PRICE DATA CHECK ===')
price_cols = ['RRP']
for col in price_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct_available = (non_null / len(df)) * 100
        if non_null > 0:
            sample_values = df[col].dropna().head(3).tolist()
            min_price = df[col].min()
            max_price = df[col].max()
            avg_price = df[col].mean()
            print(f'{col}: {non_null:,} records ({pct_available:.1f}%)')
            print(f'  Range: ${min_price:.2f} to ${max_price:.2f}, Avg: ${avg_price:.2f}')
            print(f'  Sample values: {sample_values}')
        else:
            print(f'{col}: {non_null:,} records ({pct_available:.1f}%) - ALL NULL')
    else:
        print(f'{col}: NOT FOUND')

print()
print('=== LOAD DATA CHECK ===')
load_cols = ['TOTALDEMAND', 'AVAILABLEGENERATION', 'NETINTERCHANGE']
for col in load_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct_available = (non_null / len(df)) * 100
        avg_val = df[col].mean()
        print(f'{col}: {non_null:,} records ({pct_available:.1f}%) - Avg: {avg_val:.2f} MW')

print()
print('=== SAMPLE DATA ===')
print(df.head())

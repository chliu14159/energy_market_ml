"""
Configuration file for Retail Analytics Data Preparation

This file contains all the configuration parameters for the data preparation pipeline.
"""

# File paths
INPUT_PATH = '../input'
OUTPUT_PATH = '../output'

# CSV file mappings - Focus on customer load forecasting
CSV_FILES = {
    'date_dim': 'DATE_DIM.csv',
    'training_independent': 'TRAINING_INDEPENDENT_INPUT.csv',  # Weather/external variables for modeling
    'meter_data': 'Unit_Test_Data.csv',                       # UNIT_TEST_Elanor customer load data
    'trading_price': 'TRADINGPRICE.csv',                      # Electricity market prices
}

# Column mapping for standardization
COLUMN_MAPPINGS = {
    'date_dim': {
        'source_date_col': 'ADJ_DATE',
        'target_date_col': 'DATE'
    },
    'training_independent': {
        'source_date_col': 'DATE_TIME_HH',
        'target_date_col': 'DATE_TIME'
    },
    'meter_data': {
        'source_date_col': 'DATE_TIME',
        'target_date_col': 'DATE_TIME'
    },
    'trading_price': {
        'source_date_col': 'SETTLEMENTDATE',
        'target_date_col': 'SETTLEMENTDATE'
    }
}

# Aggregation configurations
METER_DATA_AGGREGATIONS = {
    'VALUE_MW': ['sum', 'mean', 'max', 'min', 'count'],
    'NMI': 'first',
    'TRADING_INTERVAL': 'first'
}

# Weather/Training data aggregations (from TRAINING_INDEPENDENT_INPUT.csv)
TRAINING_AGGREGATIONS = {
    'AIR_TEMP': ['mean', 'max', 'min'],
    'APPARENT_TEMP': ['mean', 'max', 'min'], 
    'DEW_POINT_TEMP': ['mean', 'max', 'min'],
    'HUMIDITY': ['mean', 'max', 'min'],
    'WIND_SPEED': ['mean', 'max', 'min'],
    'PV_POWER': ['mean', 'max', 'sum'],
    'IS_WORKDAY': 'first',
    'IS_SCHOOL_HOLIDAY': 'first',
    'STATION_NAME': 'first'
}

TRADING_AGGREGATIONS = {
    'RRP': ['mean', 'max', 'min', 'std', 'count']
}

# DISPATCH_AGGREGATIONS removed - not using DISPATCHREGIONSUM for customer load forecasting
# Focus is on customer meter data + weather/training data + trading prices

# Regional focus
PRIMARY_REGION = 'QLD1'

# Output formats
OUTPUT_FORMATS = ['csv', 'parquet']

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'data_preparation.log'

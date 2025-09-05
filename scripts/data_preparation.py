#!/usr/bin/env python3
"""
Retail Analytics Platform - Data Preparation Script

This script replicates the data processing logic from the RAFA_MODULE1_UNIT_TEST notebook,
processing CSV files instead of Snowflake tables.

Usage:
    python data_preparation.py

Output:
    - Processed data files in the output/ directory
    - Both CSV and Parquet form        print(f"\n🎯 Key Outputs:")
        print(f"   • unified_analytics.parquet - Main dataset for customer load forecasting")
        print(f"   • customer_summary.csv - Daily UNIT_TEST_Elanor consumption metrics")
        print(f"   • training_summary.csv - Daily weather/training variables")
        print(f"   • trading_summary.csv - Daily electricity price data")
        print(f"   • data_quality_report.csv - Data quality metrics")
        
        print(f"\n🚀 Ready for Customer Load Forecasting!")
        print(f"   Primary dataset: pd.read_parquet('output/unified_analytics.parquet')")
        print(f"   Customer: UNIT_TEST_Elanor load patterns + weather variables + market prices")fferent use cases
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Import local modules
from config import *
from utils import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class RetailAnalyticsDataPrep:
    """Main class for retail analytics data preparation"""
    
    def __init__(self, input_path=INPUT_PATH, output_path=OUTPUT_PATH):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized data preparation")
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        
        # Data containers
        self.raw_data = {}
        self.processed_data = {}
        
    def load_csv_files(self):
        """Load all CSV files from input directory"""
        logger.info("Loading CSV files...")
        
        for key, filename in CSV_FILES.items():
            file_path = self.input_path / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.raw_data[key] = df
                    
                    # Print profile
                    profile = profile_dataframe(df, filename)
                    print_dataframe_profile(profile)
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"File not found: {filename}")
        
        return self.raw_data
    
    def standardize_datetime_columns(self):
        """Standardize datetime columns across all datasets"""
        logger.info("Standardizing datetime columns...")
        
        # Date dimension
        if 'date_dim' in self.raw_data:
            df = self.raw_data['date_dim'].copy()
            df = standardize_date_column(df, 'ADJ_DATE', 'DATE')
            self.processed_data['date_dim'] = df
        
        # Training independent variables (weather/external data)
        if 'training_independent' in self.raw_data:
            df = self.raw_data['training_independent'].copy()
            df = standardize_date_column(df, 'DATE_TIME_HH', 'DATE_TIME')
            df['DATE'] = df['DATE_TIME'].dt.date
            df['DATE'] = pd.to_datetime(df['DATE'])
            self.processed_data['training_independent'] = df
        
        # Meter data (UNIT_TEST_Elanor customer data)
        if 'meter_data' in self.raw_data:
            df = self.raw_data['meter_data'].copy()
            df = standardize_date_column(df, 'DATE_TIME', 'DATE_TIME')
            df['DATE'] = df['DATE_TIME'].dt.date
            df['DATE'] = pd.to_datetime(df['DATE'])
            self.processed_data['meter_data'] = df
        
        # Trading price
        if 'trading_price' in self.raw_data:
            df = self.raw_data['trading_price'].copy()
            df = standardize_date_column(df, 'SETTLEMENTDATE', 'SETTLEMENTDATE')
            df['DATE'] = df['SETTLEMENTDATE'].dt.date
            df['DATE'] = pd.to_datetime(df['DATE'])
            self.processed_data['trading_price'] = df
    
    def create_customer_meterdata_summary(self):
        """
        Replicate the CUSTOMER_METERDATA_SUMMARY logic:
        - Aggregate meter data by customer, date, and meter suffix
        - Calculate daily statistics
        """
        logger.info("Creating customer meter data summary...")
        
        if 'meter_data' not in self.processed_data:
            logger.warning("No meter data available for summary")
            return None
        
        df = self.processed_data['meter_data'].copy()
        
        # Use utility function for aggregation
        summary = aggregate_to_daily(
            df, 
            ['CUSTOMER_NAME', 'METER_ID_SUFFIX'], 
            METER_DATA_AGGREGATIONS
        )
        
        # Rename columns for clarity
        column_renames = {
            'VALUE_MW_sum': 'DAILY_TOTAL_MW',
            'VALUE_MW_mean': 'DAILY_MEAN_MW', 
            'VALUE_MW_max': 'DAILY_MAX_MW',
            'VALUE_MW_min': 'DAILY_MIN_MW',
            'VALUE_MW_count': 'INTERVALS_COUNT',
            'NMI_first': 'NMI',
            'TRADING_INTERVAL_first': 'TRADING_INTERVAL'
        }
        
        summary.rename(columns=column_renames, inplace=True)
        
        self.processed_data['customer_summary'] = summary
        
        # Print profile
        profile = profile_dataframe(summary, "Customer Meter Summary")
        print_dataframe_profile(profile)
        
        return summary
    
    def create_market_data_summary(self):
        """
        Create daily market data summary from trading prices
        """
        logger.info("Creating market data summary...")
        
        summaries = {}
        
        # Trading price daily summary
        if 'trading_price' in self.processed_data:
            trading_df = self.processed_data['trading_price'].copy()
            
            trading_summary = aggregate_to_daily(
                trading_df,
                ['REGIONID'],
                TRADING_AGGREGATIONS
            )
            
            summaries['trading_summary'] = trading_summary
            
            # Print profile
            profile = profile_dataframe(trading_summary, "Trading Price Summary")
            print_dataframe_profile(profile)
        
        self.processed_data.update(summaries)
        return summaries
    
    def create_training_data_summary(self):
        """
        Create daily training/weather data summary from TRAINING_INDEPENDENT_INPUT
        """
        logger.info("Creating training data summary...")
        
        if 'training_independent' not in self.processed_data:
            logger.warning("No training independent data available")
            return None
        
        df = self.processed_data['training_independent'].copy()
        
        # Use utility function for aggregation
        training_summary = aggregate_to_daily(
            df,
            [],  # No additional grouping columns beyond DATE
            TRAINING_AGGREGATIONS
        )
        
        self.processed_data['training_summary'] = training_summary
        
        # Print profile
        profile = profile_dataframe(training_summary, "Training Data Summary")
        print_dataframe_profile(profile)
        
        return training_summary
    
    def create_unified_analytics_dataset(self):
        """
        Create the main unified analytics dataset by joining all summaries
        Similar to the final analytics table in the UNIT_TEST notebook
        """
        logger.info("Creating unified analytics dataset...")
        
        # Start with date dimension as spine
        if 'date_dim' not in self.processed_data:
            logger.error("Date dimension required for unified dataset")
            return None
        
        analytics_df = self.processed_data['date_dim'][['DATE', 'DAY_TYPE_QLD']].copy()
        logger.info(f"Starting with date spine: {analytics_df.shape}")
        
        # Add training data
        if 'training_summary' in self.processed_data:
            analytics_df = merge_with_validation(
                analytics_df, 
                self.processed_data['training_summary'], 
                on='DATE'
            )
        
        # Add customer data (this will multiply rows if multiple customers)
        if 'customer_summary' in self.processed_data:
            analytics_df = merge_with_validation(
                analytics_df,
                self.processed_data['customer_summary'],
                on='DATE'
            )
        
        # Add trading data (focus on primary region)
        if 'trading_summary' in self.processed_data:
            trading_filtered = filter_by_region(
                self.processed_data['trading_summary'], 
                'REGIONID', 
                PRIMARY_REGION
            ).drop('REGIONID', axis=1)
            
            analytics_df = merge_with_validation(
                analytics_df,
                trading_filtered,
                on='DATE'
            )
        
        # Add dispatch data (focus on primary region) - REMOVED for customer load forecasting
        # if 'dispatch_summary' in self.processed_data:
        #     dispatch_filtered = filter_by_region(
        #         self.processed_data['dispatch_summary'],
        #         'REGIONID',
        #         PRIMARY_REGION
        #     ).drop('REGIONID', axis=1)
        #     
        #     analytics_df = merge_with_validation(
        #         analytics_df,
        #         dispatch_filtered,
        #         on='DATE'
        #     )
        
        self.processed_data['unified_analytics'] = analytics_df
        
        # Print profile
        profile = profile_dataframe(analytics_df, "Unified Analytics Dataset")
        print_dataframe_profile(profile)
        
        return analytics_df
    
    def save_outputs(self, formats=OUTPUT_FORMATS):
        """Save processed datasets in specified formats"""
        logger.info(f"Saving outputs in formats: {formats}")
        
        for dataset_name, df in self.processed_data.items():
            output_path = self.output_path / dataset_name
            save_dataframe(df, output_path, formats)
    
    def generate_data_quality_report(self):
        """Generate and save data quality report"""
        logger.info("Generating data quality report...")
        
        quality_report = create_data_quality_report(self.processed_data)
        
        # Save the report
        report_path = self.output_path / "data_quality_report"
        save_dataframe(quality_report, report_path, ['csv'])
        
        # Print the report
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(quality_report.to_string(index=False))
        print("="*60)
        
        return quality_report
    
    def run_full_pipeline(self):
        """Execute the complete data preparation pipeline"""
        logger.info("Starting full data preparation pipeline...")
        
        try:
            # Load data
            self.load_csv_files()
            
            # Process datetime columns
            self.standardize_datetime_columns()
            
            # Create summaries
            self.create_customer_meterdata_summary()
            self.create_market_data_summary()
            self.create_training_data_summary()
            
            # Create unified dataset
            self.create_unified_analytics_dataset()
            
            # Generate quality report
            self.generate_data_quality_report()
            
            # Save outputs
            self.save_outputs()
            
            logger.info("Data preparation pipeline completed successfully!")
            
            # Print final summary
            self.print_final_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def print_final_summary(self):
        """Print final summary of the pipeline execution"""
        print("\n" + "="*70)
        print("RETAIL ANALYTICS DATA PREPARATION - FINAL SUMMARY")
        print("="*70)
        
        print(f"\n📁 Input Directory: {self.input_path}")
        print(f"💾 Output Directory: {self.output_path}")
        
        print(f"\n📊 Processed Datasets: {len(self.processed_data)}")
        for dataset_name, df in self.processed_data.items():
            print(f"   • {dataset_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        print(f"\n🎯 Key Outputs:")
        print(f"   • unified_analytics.parquet - Main dataset for Streamlit app")
        print(f"   • customer_summary.csv - Daily customer consumption metrics")
        print(f"   • weather_summary.csv - Daily weather aggregations")
        print(f"   • trading_summary.csv - Daily electricity price data")
        print(f"   • dispatch_summary.csv - Daily grid dispatch data")
        print(f"   • data_quality_report.csv - Data quality metrics")
        
        print(f"\n� Ready for Streamlit Application!")
        print(f"   Primary dataset: pd.read_parquet('output/unified_analytics.parquet')")
        
        print("="*70)


def main():
    """Main execution function"""
    print("🏪 Starting Customer Load Forecasting - Data Preparation")
    print("="*60)
    
    # Initialize data processor
    processor = RetailAnalyticsDataPrep()
    
    # Run full pipeline
    processor.run_full_pipeline()
    
    print("\n✅ Customer load forecasting data preparation completed successfully!")
    print("    Ready for UNIT_TEST_Elanor load modeling and prediction!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Half-Hourly Energy Market Data Preparation Script

This script processes CSV files for half-hourly energy market analysis,
preserving the 48 periods per day structure used in NEM markets.

Key Changes from Daily Version:
- Preserves PERIOD_HH (1-48) instead of aggregating to daily
- Uses DATE_TIME_HH as primary temporal key
- Maintains half-hourly granularity for load, weather, and price data
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

class HalfHourlyDataProcessor:
    """
    Half-hourly data processing for energy market analytics
    """
    
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
        self.analytics_data = None
        
    def load_raw_data(self):
        """
        Load all CSV files into memory with proper datetime handling
        """
        logger.info("Loading raw data files...")
        
        # Convert string paths to Path objects
        input_path = Path(INPUT_PATH)
        
        # Load DATE_TIME_DIM (our new half-hourly dimension)
        if (input_path / 'DATE_TIME_DIM.csv').exists():
            logger.info("Loading DATE_TIME_DIM...")
            df = pd.read_csv(input_path / 'DATE_TIME_DIM.csv')
            df['DATE_TIME_HH'] = pd.to_datetime(df['DATE_TIME_HH'])
            df['DATE'] = pd.to_datetime(df['DATE'])
            self.raw_data['datetime_dim'] = df
            logger.info(f"   Loaded {len(df):,} datetime records")
        
        # Load TRAINING_INDEPENDENT_INPUT (weather data)
        if (input_path / 'TRAINING_INDEPENDENT_INPUT.csv').exists():
            logger.info("Loading TRAINING_INDEPENDENT_INPUT...")
            df = pd.read_csv(input_path / 'TRAINING_INDEPENDENT_INPUT.csv')
            df['DATE_TIME_HH'] = pd.to_datetime(df['DATE_TIME_HH'])
            # Extract date for joining with datetime dimension
            df['DATE'] = df['DATE_TIME_HH'].dt.date
            df['DATE'] = pd.to_datetime(df['DATE'])
            self.raw_data['training_independent'] = df
            logger.info(f"   Loaded {len(df):,} weather records")
        
        # Load Unit_Test_Data (customer meter data)
        if (input_path / 'Unit_Test_Data.csv').exists():
            logger.info("Loading Unit_Test_Data...")
            df = pd.read_csv(input_path / 'Unit_Test_Data.csv')
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
            # Extract date and period from datetime
            df['DATE'] = df['DATE_TIME'].dt.date
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Calculate PERIOD_HH from time (align with NEM convention)
            # NEM: Period 1 ends at 00:30, Period 2 ends at 01:00, etc.
            df['HOUR'] = df['DATE_TIME'].dt.hour
            df['MINUTE'] = df['DATE_TIME'].dt.minute
            df['PERIOD_HH'] = ((df['HOUR'] * 60 + df['MINUTE']) // 30) 
            # Adjust for NEM convention where period represents the end time
            df['PERIOD_HH'] = df['PERIOD_HH'] + 1
            # Handle midnight case (00:00 should be period 48 of previous day)
            df.loc[df['PERIOD_HH'] == 49, 'PERIOD_HH'] = 1
            
            # Filter for UNIT_TEST_Elanor customer
            df = df[df['CUSTOMER_NAME'] == 'UNIT_TEST_Elanor'].copy()
            
            self.raw_data['meter_data'] = df
            logger.info(f"   Loaded {len(df):,} meter readings for UNIT_TEST_Elanor")
        
        # Load TRADINGPRICE (electricity market prices)
        if (input_path / 'TRADINGPRICE.csv').exists():
            logger.info("Loading TRADINGPRICE...")
            df = pd.read_csv(input_path / 'TRADINGPRICE.csv')
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
            # Extract date and period
            df['DATE'] = df['SETTLEMENTDATE'].dt.date  
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['HOUR'] = df['SETTLEMENTDATE'].dt.hour
            df['MINUTE'] = df['SETTLEMENTDATE'].dt.minute
            df['PERIOD_HH'] = ((df['HOUR'] * 60 + df['MINUTE']) // 30)
            # Adjust for NEM convention where period represents the end time  
            df['PERIOD_HH'] = df['PERIOD_HH'] + 1
            # Handle midnight case (00:00 should be period 48 of previous day)
            df.loc[df['PERIOD_HH'] == 49, 'PERIOD_HH'] = 1
            
            self.raw_data['trading_price'] = df
            logger.info(f"   Loaded {len(df):,} price records")
    
    def create_halfhourly_customer_summary(self):
        """
        Create half-hourly customer meter data summary preserving PERIOD_HH
        """
        logger.info("Creating half-hourly customer summary...")
        
        if 'meter_data' not in self.raw_data:
            logger.warning("No meter data available")
            return None
        
        df = self.raw_data['meter_data'].copy()
        
        # Group by customer, date, and period to preserve half-hourly structure
        summary = df.groupby(['CUSTOMER_NAME', 'DATE', 'PERIOD_HH', 'METER_ID_SUFFIX']).agg({
            'VALUE_MW': ['sum', 'mean', 'max', 'min', 'count'],
            'DATE_TIME': 'first'  # Keep the datetime for reference
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'CUSTOMER_NAME', 'DATE', 'PERIOD_HH', 'METER_ID_SUFFIX',
            'VALUE_MW_sum', 'VALUE_MW_mean', 'VALUE_MW_max', 'VALUE_MW_min', 'VALUE_MW_count',
            'DATE_TIME_HH'
        ]
        
        # Rename for clarity
        summary = summary.rename(columns={
            'VALUE_MW_sum': 'HALFHOURLY_TOTAL_MW',
            'VALUE_MW_mean': 'HALFHOURLY_MEAN_MW',
            'VALUE_MW_max': 'HALFHOURLY_MAX_MW',
            'VALUE_MW_min': 'HALFHOURLY_MIN_MW',
            'VALUE_MW_count': 'READING_COUNT'
        })
        
        # Pivot to get separate columns for B1 and E1 meter suffixes
        summary_pivot = summary.pivot_table(
            index=['CUSTOMER_NAME', 'DATE', 'PERIOD_HH', 'DATE_TIME_HH'],
            columns='METER_ID_SUFFIX',
            values=['HALFHOURLY_TOTAL_MW', 'HALFHOURLY_MEAN_MW'],
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        summary_pivot.columns = [
            col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
            for col in summary_pivot.columns
        ]
        
        # Calculate net consumption (E1 - B1 for import/export)
        if 'HALFHOURLY_TOTAL_MW_E1' in summary_pivot.columns and 'HALFHOURLY_TOTAL_MW_B1' in summary_pivot.columns:
            summary_pivot['NET_CONSUMPTION_MW'] = (
                summary_pivot['HALFHOURLY_TOTAL_MW_E1'] - summary_pivot['HALFHOURLY_TOTAL_MW_B1']
            )
        
        self.processed_data['customer_summary'] = summary_pivot
        logger.info(f"   Created {len(summary_pivot):,} half-hourly customer records")
        
        return summary_pivot
    
    def create_halfhourly_weather_summary(self):
        """
        Create half-hourly weather summary preserving PERIOD_HH
        """
        logger.info("Creating half-hourly weather summary...")
        
        if 'training_independent' not in self.raw_data:
            logger.warning("No weather data available")
            return None
        
        df = self.raw_data['training_independent'].copy()
        
        # Group by date and period to preserve half-hourly structure
        weather_cols = [col for col in df.columns if col not in ['STATION_NAME', 'DATE', 'PERIOD_HH', 'DATE_TIME_HH', 'YEAR', 'MONTH', 'DAY_TYPE']]
        
        # Aggregate weather data by DATE and PERIOD_HH
        summary = df.groupby(['DATE', 'PERIOD_HH']).agg({
            **{col: 'mean' for col in weather_cols},
            'DATE_TIME_HH': 'first',
            'DAY_TYPE': 'first',
            'STATION_NAME': 'first'
        }).reset_index()
        
        # Add suffix to weather columns for clarity
        weather_rename = {col: f"{col}_mean" for col in weather_cols}
        weather_rename.update({
            'DAY_TYPE': 'DAY_TYPE_first',
            'STATION_NAME': 'STATION_NAME_first'
        })
        summary = summary.rename(columns=weather_rename)
        
        self.processed_data['weather_summary'] = summary
        logger.info(f"   Created {len(summary):,} half-hourly weather records")
        
        return summary
    
    def create_halfhourly_price_summary(self):
        """
        Create half-hourly electricity price summary preserving PERIOD_HH
        """
        logger.info("Creating half-hourly price summary...")
        
        if 'trading_price' not in self.raw_data:
            logger.warning("No trading price data available")
            return None
        
        df = self.raw_data['trading_price'].copy()
        
        # Group by date and period for half-hourly prices
        summary = df.groupby(['DATE', 'PERIOD_HH']).agg({
            'RRP': ['mean', 'max', 'min', 'std'],
            'SETTLEMENTDATE': 'first'
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'DATE', 'PERIOD_HH',
            'RRP_mean', 'RRP_max', 'RRP_min', 'RRP_std',
            'SETTLEMENT_DATETIME'
        ]
        
        # Fill NaN in std with 0
        summary['RRP_std'] = summary['RRP_std'].fillna(0)
        
        self.processed_data['price_summary'] = summary
        logger.info(f"   Created {len(summary):,} half-hourly price records")
        
        return summary
    
    def create_unified_analytics_dataset(self):
        """
        Create unified half-hourly analytics dataset
        """
        logger.info("Creating unified half-hourly analytics dataset...")
        
        # Start with datetime dimension as base
        if 'datetime_dim' not in self.raw_data:
            logger.error("DATE_TIME_DIM not available - cannot create unified dataset")
            return None
        
        base_df = self.raw_data['datetime_dim'][['DATE', 'PERIOD_HH', 'DATE_TIME_HH', 
                                                'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK',
                                                'IS_WEEKEND', 'IS_WEEKDAY', 'IS_PEAK_PERIOD',
                                                'IS_OFF_PEAK_PERIOD', 'TIME_CATEGORY', 'DEMAND_PERIOD',
                                                'IS_PUBLIC_HOLIDAY']].copy()
        
        # Join customer data
        if 'customer_summary' in self.processed_data:
            logger.info("   Joining customer data...")
            base_df = base_df.merge(
                self.processed_data['customer_summary'],
                on=['DATE', 'PERIOD_HH'],
                how='left'
            )
        
        # Join weather data
        if 'weather_summary' in self.processed_data:
            logger.info("   Joining weather data...")
            base_df = base_df.merge(
                self.processed_data['weather_summary'],
                on=['DATE', 'PERIOD_HH'],
                how='left'
            )
        
        # Join price data
        if 'price_summary' in self.processed_data:
            logger.info("   Joining price data...")
            base_df = base_df.merge(
                self.processed_data['price_summary'],
                on=['DATE', 'PERIOD_HH'],
                how='left'
            )
        
        # Filter to dates where we have customer data
        if 'CUSTOMER_NAME' in base_df.columns:
            date_range = base_df[base_df['CUSTOMER_NAME'].notna()]['DATE']
            if not date_range.empty:
                min_date, max_date = date_range.min(), date_range.max()
                base_df = base_df[(base_df['DATE'] >= min_date) & (base_df['DATE'] <= max_date)]
                logger.info(f"   Filtered to date range: {min_date} to {max_date}")
        
        # Add derived features for half-hourly analysis
        if 'NET_CONSUMPTION_MW' in base_df.columns:
            # Create load categories
            base_df['LOAD_CATEGORY'] = pd.cut(
                base_df['NET_CONSUMPTION_MW'].fillna(0),
                bins=[-np.inf, 0, 0.05, 0.1, np.inf],
                labels=['Zero/Export', 'Low', 'Medium', 'High']
            )
        
        self.analytics_data = base_df
        logger.info(f"   Created unified dataset with {len(base_df):,} half-hourly records")
        logger.info(f"   Columns: {len(base_df.columns)}")
        
        return base_df
    
    def save_processed_data(self):
        """
        Save all processed datasets
        """
        logger.info("Saving processed datasets...")
        
        # Ensure output directory exists
        output_path = Path(OUTPUT_PATH)
        output_path.mkdir(exist_ok=True)
        
        # Save individual summaries
        if 'customer_summary' in self.processed_data:
            self.processed_data['customer_summary'].to_csv(
                output_path / 'halfhourly_customer_summary.csv', index=False
            )
            logger.info(f"   Saved halfhourly_customer_summary.csv")
        
        if 'weather_summary' in self.processed_data:
            self.processed_data['weather_summary'].to_csv(
                output_path / 'halfhourly_weather_summary.csv', index=False
            )
            logger.info(f"   Saved halfhourly_weather_summary.csv")
        
        if 'price_summary' in self.processed_data:
            self.processed_data['price_summary'].to_csv(
                output_path / 'halfhourly_price_summary.csv', index=False
            )
            logger.info(f"   Saved halfhourly_price_summary.csv")
        
        # Save unified analytics dataset
        if self.analytics_data is not None:
            # Save as both CSV and Parquet
            self.analytics_data.to_csv(
                output_path / 'unified_analytics_halfhourly.csv', index=False
            )
            self.analytics_data.to_parquet(
                output_path / 'unified_analytics_halfhourly.parquet', index=False
            )
            logger.info(f"   Saved unified_analytics_halfhourly.csv and .parquet")
            
            # Also save as the main file (overwrites daily version)
            self.analytics_data.to_parquet(
                output_path / 'unified_analytics.parquet', index=False
            )
            logger.info(f"   Updated unified_analytics.parquet with half-hourly data")
    
    def generate_data_quality_report(self):
        """
        Generate data quality report for half-hourly data
        """
        logger.info("Generating data quality report...")
        
        if self.analytics_data is None:
            logger.warning("No analytics data available for quality report")
            return None
        
        df = self.analytics_data
        
        quality_metrics = []
        
        # Overall statistics
        quality_metrics.append({
            'Metric': 'Total Records',
            'Value': len(df),
            'Description': 'Total half-hourly records in unified dataset'
        })
        
        quality_metrics.append({
            'Metric': 'Date Range',
            'Value': f"{df['DATE'].min()} to {df['DATE'].max()}",
            'Description': 'Complete date range covered'
        })
        
        quality_metrics.append({
            'Metric': 'Period Range',
            'Value': f"{df['PERIOD_HH'].min()} to {df['PERIOD_HH'].max()}",
            'Description': 'Half-hourly periods covered (1-48)'
        })
        
        # Customer data quality
        if 'NET_CONSUMPTION_MW' in df.columns:
            customer_data = df[df['NET_CONSUMPTION_MW'].notna()]
            quality_metrics.append({
                'Metric': 'Customer Data Coverage',
                'Value': f"{len(customer_data):,} records ({len(customer_data)/len(df)*100:.1f}%)",
                'Description': 'Records with customer load data'
            })
            
            quality_metrics.append({
                'Metric': 'Average Half-Hourly Load',
                'Value': f"{customer_data['NET_CONSUMPTION_MW'].mean():.4f} MW",
                'Description': 'Mean consumption across all periods'
            })
        
        # Weather data quality
        weather_cols = [col for col in df.columns if '_mean' in col and any(x in col for x in ['TEMP', 'HUMID', 'WIND'])]
        if weather_cols:
            weather_coverage = df[weather_cols[0]].notna().sum()
            quality_metrics.append({
                'Metric': 'Weather Data Coverage',
                'Value': f"{weather_coverage:,} records ({weather_coverage/len(df)*100:.1f}%)",
                'Description': 'Records with weather data'
            })
        
        # Price data quality
        if 'RRP_mean' in df.columns:
            price_coverage = df['RRP_mean'].notna().sum()
            quality_metrics.append({
                'Metric': 'Price Data Coverage',
                'Value': f"{price_coverage:,} records ({price_coverage/len(df)*100:.1f}%)",
                'Description': 'Records with electricity price data'
            })
            
            quality_metrics.append({
                'Metric': 'Average Half-Hourly Price',
                'Value': f"${df['RRP_mean'].mean():.2f}/MWh",
                'Description': 'Mean electricity price across all periods'
            })
        
        # Peak vs Off-Peak distribution
        if 'DEMAND_PERIOD' in df.columns:
            period_dist = df['DEMAND_PERIOD'].value_counts()
            for period, count in period_dist.items():
                quality_metrics.append({
                    'Metric': f'{period} Periods',
                    'Value': f"{count:,} records ({count/len(df)*100:.1f}%)",
                    'Description': f'Half-hourly records in {period} demand period'
                })
        
        quality_df = pd.DataFrame(quality_metrics)
        output_path = Path(OUTPUT_PATH)
        quality_df.to_csv(output_path / 'halfhourly_data_quality_report.csv', index=False)
        
        logger.info(f"   Generated quality report with {len(quality_metrics)} metrics")
        
        return quality_df
    
    def run_full_pipeline(self):
        """
        Execute the complete half-hourly data processing pipeline
        """
        start_time = datetime.now()
        
        logger.info("🚀 Starting Half-Hourly Energy Market Data Processing Pipeline")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load raw data
            self.load_raw_data()
            
            # Step 2: Create summaries
            self.create_halfhourly_customer_summary()
            self.create_halfhourly_weather_summary()
            self.create_halfhourly_price_summary()
            
            # Step 3: Create unified dataset
            self.create_unified_analytics_dataset()
            
            # Step 4: Save processed data
            self.save_processed_data()
            
            # Step 5: Generate quality report
            quality_report = self.generate_data_quality_report()
            
            # Completion summary
            processing_time = datetime.now() - start_time
            
            logger.info("=" * 70)
            logger.info("✅ Half-Hourly Processing Pipeline Complete!")
            logger.info(f"   Processing time: {processing_time.total_seconds():.2f} seconds")
            
            if self.analytics_data is not None:
                logger.info(f"   Final dataset: {len(self.analytics_data):,} half-hourly records")
                logger.info(f"   Date range: {self.analytics_data['DATE'].min()} to {self.analytics_data['DATE'].max()}")
                logger.info(f"   Features: {self.analytics_data.shape[1]} columns")
            
            logger.info("\n🎯 Key Outputs:")
            logger.info("   • unified_analytics.parquet - Main half-hourly dataset")
            logger.info("   • halfhourly_customer_summary.csv - Half-hourly customer loads")
            logger.info("   • halfhourly_weather_summary.csv - Half-hourly weather data") 
            logger.info("   • halfhourly_price_summary.csv - Half-hourly electricity prices")
            logger.info("   • halfhourly_data_quality_report.csv - Data quality metrics")
            
            logger.info("\n🚀 Ready for Half-Hourly Load Forecasting!")
            logger.info("   Primary dataset: pd.read_parquet('output/unified_analytics.parquet')")
            logger.info("   Customer: UNIT_TEST_Elanor half-hourly patterns + weather + prices")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """
    Main execution function
    """
    processor = HalfHourlyDataProcessor()
    processor.run_full_pipeline()

if __name__ == "__main__":
    main()

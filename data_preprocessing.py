import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_path: str, date_format: str = '%d-%m-%y'):
        self.data_path = data_path
        self.date_format = date_format
        self.df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            self.df = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Raw data loaded successfully. Shape: {self.df.shape}, Columns: {list(self.df.columns)}")
            
            expected_dtypes = {
                'is_canceled': 'int8',
                'lead_time': 'int16',
                'adults': 'int8',
                'children': 'float32',  
                'babies': 'int8',
                'adr': 'float32'
            }
            missing_cols = [col for col in expected_dtypes if col not in self.df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in CSV: {missing_cols}. Adjusting dtype application")
            
            available_dtypes = {col: dtype for col, dtype in expected_dtypes.items() if col in self.df.columns}
            for col, dtype in available_dtypes.items():
                try:
                    original_series = self.df[col]
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    invalid_rows = original_series[self.df[col].isna() & original_series.notna()]
                    if not invalid_rows.empty:
                        logger.warning(f"Column '{col}' has non-numeric values coerced to NaN: {invalid_rows.head().to_dict()}")
                    self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    logger.error(f"Failed to convert column '{col}' to {dtype}: {e}", exc_info=True)
                    return None
            
            logger.info(f"Data loaded with types applied. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return None
            
    def clean_data(self) -> Optional[pd.DataFrame]:
        if self.df is None:
            self.load_data()
        if self.df is None:
            return None
            
        self.processed_df = self.df.copy()
        fill_values = {'children': 0, 'agent': 0, 'company': 0}
        self.processed_df.fillna(fill_values, inplace=True)
        int_cols = ['children', 'agent', 'company']
        
        for col in int_cols:
            if col in self.processed_df.columns:
                self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce').astype('int32')
        
        if all(col in self.processed_df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            self.processed_df['arrival_date'] = pd.to_datetime(
                self.processed_df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']]
                .astype(str).agg('-'.join, axis=1), format='%Y-%B-%d', errors='coerce')
        else:
            logger.warning("Date columns missing skipping arrival_date creation")
        
        if all(col in self.processed_df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            self.processed_df['total_stays'] = (self.processed_df['stays_in_weekend_nights'] + 
                                               self.processed_df['stays_in_week_nights']).astype('int16')
        if all(col in self.processed_df.columns for col in ['adults', 'children', 'babies']):
            self.processed_df['total_guests'] = (self.processed_df['adults'] + 
                                                self.processed_df['children'] + 
                                                self.processed_df['babies']).astype('int16')
        if all(col in self.processed_df.columns for col in ['adr', 'total_stays']):
            self.processed_df['total_revenue'] = (self.processed_df['adr'] * 
                                                 self.processed_df['total_stays']).astype('float32')
        
        if 'reservation_status_date' in self.processed_df.columns:
            self.processed_df['reservation_status_date'] = pd.to_datetime(
                self.processed_df['reservation_status_date'], format=self.date_format, errors='coerce')
        

        if 'adr' in self.processed_df.columns:
            q1, q3 = self.processed_df['adr'].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = self.processed_df[(self.processed_df['adr'] < lower_bound) | (self.processed_df['adr'] > upper_bound)]
            if not outliers.empty:
                logger.info(f"Removed {len(outliers)} outliers from 'adr' (bounds: {lower_bound}, {upper_bound})")
                self.processed_df = self.processed_df[(self.processed_df['adr'] >= lower_bound) & 
                                                     (self.processed_df['adr'] <= upper_bound)]
        

        if 'adults' in self.processed_df.columns and 'adr' in self.processed_df.columns:
            self.processed_df = self.processed_df.query('adults > 0 and adr >= 0')
        
        self.processed_df['booking_id'] = np.arange(len(self.processed_df), dtype='int32')
        
        logger.info(f"Data cleaning completed. Processed shape: {self.processed_df.shape}")
        return self.processed_df
    
    def _add_time_features(self) -> None:
        if 'arrival_date' in self.processed_df.columns:
            self.processed_df['year_month'] = self.processed_df['arrival_date'].dt.strftime('%Y-%m')
            month_to_season = {
                'January': 'Winter', 'February': 'Winter', 'December': 'Winter',
                'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
                'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
                'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
            }
            if 'arrival_date_month' in self.processed_df.columns:
                self.processed_df['season'] = self.processed_df['arrival_date_month'].map(month_to_season)
    
    def _add_categorical_features(self) -> None:
        if 'lead_time' in self.processed_df.columns:
            self.processed_df['lead_time_category'] = pd.cut(
                self.processed_df['lead_time'], bins=[0, 7, 30, 90, 180, 365, float('inf')],
                labels=['Last Minute', 'Short', 'Medium', 'Long', 'Very Long', 'Extreme'],
                right=False).astype('category')
        if 'total_stays' in self.processed_df.columns:
            self.processed_df['stay_length_category'] = pd.cut(
                self.processed_df['total_stays'], bins=[0, 1, 3, 7, 14, float('inf')],
                labels=['1 Night', '2-3 Nights', '4-7 Nights', '8-14 Nights', '15+ Nights'],
                right=False).astype('category')
        if 'adr' in self.processed_df.columns:
            self.processed_df['price_category'] = pd.qcut(
                self.processed_df['adr'], q=5, labels=['Budget', 'Economy', 'Average', 'Premium', 'Luxury'],
                duplicates='drop').astype('category')
    
    def create_analytics_features(self) -> Optional[pd.DataFrame]:
        if self.processed_df is None:
            self.clean_data()
        if self.processed_df is None:
            return None
        self._add_time_features()
        self._add_categorical_features()
        logger.info("Analytics features created successfully")
        return self.processed_df
    
    def save_processed_data(self, output_path: str = "processed_hotel_bookings.csv.gz") -> bool:
        if self.processed_df is None:
            self.create_analytics_features()
        if self.processed_df is None:
            return False
        try:
            self.processed_df.to_csv(output_path, index=False, compression='gzip')
            logger.info(f"Processed data saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        if self.processed_df is None:
            self.create_analytics_features()
        return self.processed_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor("data/hotel_bookings.csv")
    df = preprocessor.load_data()
    if df is not None:
        processed_df = preprocessor.create_analytics_features()
        preprocessor.save_processed_data()
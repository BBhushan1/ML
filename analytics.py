import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from pathlib import Path
import logging
from typing import Optional, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelAnalytics:
    def __init__(self, df: Optional[pd.DataFrame] = None, data_path: Optional[str] = None):
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path, parse_dates=['arrival_date', 'reservation_status_date'])
            logger.info(f"Data loaded from {data_path}. Shape: {self.df.shape}")
        else:
            raise ValueError("data_path must be provided")

        self.output_dir = Path("analytics_output")
        self.output_dir.mkdir(exist_ok=True)
        self.analytics_results: Dict[str, Any] = {}
        self.plotly_enabled = False
        
    def _save_plot(self, fig, filename: str, use_plotly: bool = False) -> None:
        if use_plotly and self.plotly_enabled:
            fig.write_html(self.output_dir / f"{filename}.html")
        else:
            fig.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def revenue_trends(self, save_fig: bool = True) -> Dict[str, Any]:
        monthly_revenue = self.df.groupby('year_month')['total_revenue'].sum().reset_index()
        monthly_revenue['year_month'] = pd.to_datetime(monthly_revenue['year_month'] + '-01')
        hotel_revenue = self.df.groupby(['year_month', 'hotel'])['total_revenue'].sum().unstack().fillna(0)
        
        if save_fig:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=monthly_revenue, x='year_month', y='total_revenue', marker='o')
            plt.title('Monthly Revenue Trend', fontsize=14)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Total Revenue', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "monthly_revenue_trend")
            
            if self.plotly_enabled:
                fig = px.line(hotel_revenue.reset_index(), x='year_month', y=hotel_revenue.columns,
                             title='Monthly Revenue by Hotel Type', labels={'value': 'Revenue', 'variable': 'Hotel'})
                self._save_plot(fig, "revenue_by_hotel_type", use_plotly=True)
            else:
                plt.figure(figsize=(12, 6))
                hotel_revenue.plot(marker='o', ax=plt.gca())
                plt.title('Monthly Revenue by Hotel Type', fontsize=14)
                plt.xlabel('Month', fontsize=12)
                plt.ylabel('Total Revenue', fontsize=12)
                plt.legend(title='Hotel')
                plt.grid(True, alpha=0.3)
                self._save_plot(plt.gcf(), "revenue_by_hotel_type")
        
        result = {
            'monthly_revenue': monthly_revenue.to_dict('records'),
            'hotel_revenue': hotel_revenue.reset_index().to_dict('records')
        }
        self.analytics_results['revenue_trends'] = result
        return result
    
    def cancellation_analysis(self, save_fig: bool = True) -> Dict[str, Any]:
        cancellation_rate = self.df['is_canceled'].mean() * 100
        monthly_cancellation = self.df.groupby('year_month').agg(
            total_bookings=pd.NamedAgg(column='booking_id', aggfunc='count'),
            cancelled_bookings=pd.NamedAgg(column='is_canceled', aggfunc='sum')
        ).assign(cancellation_rate=lambda x: x['cancelled_bookings'] / x['total_bookings'] * 100)
        monthly_cancellation.index = pd.to_datetime(monthly_cancellation.index + '-01')
        
        hotel_cancellation = self.df.groupby('hotel')['is_canceled'].agg(['count', 'sum', 'mean'])
        hotel_cancellation.columns = ['total_bookings', 'cancelled_bookings', 'cancellation_rate']
        hotel_cancellation['cancellation_rate'] *= 100
        
        if save_fig:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=monthly_cancellation, x=monthly_cancellation.index, y='cancellation_rate', marker='o')
            plt.title('Monthly Cancellation Rate', fontsize=14)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Cancellation Rate (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "monthly_cancellation_rate")
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=hotel_cancellation.reset_index(), x='hotel', y='cancellation_rate')
            plt.title('Cancellation Rate by Hotel Type', fontsize=14)
            plt.xlabel('Hotel Type', fontsize=12)
            plt.ylabel('Cancellation Rate (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "cancellation_rate_by_hotel")
        
        result = {
            'overall_cancellation_rate': cancellation_rate,
            'monthly_cancellation': monthly_cancellation.reset_index().to_dict('records'),
            'hotel_cancellation': hotel_cancellation.reset_index().to_dict('records')
        }
        self.analytics_results['cancellation_analysis'] = result
        return result
    
    def geographical_distribution(self, save_fig: bool = True) -> Dict[str, Any]:
        country_bookings = self.df.groupby('country').agg(
            total_bookings=pd.NamedAgg(column='booking_id', aggfunc='count'),
            cancelled_bookings=pd.NamedAgg(column='is_canceled', aggfunc='sum'),
            total_revenue=pd.NamedAgg(column='total_revenue', aggfunc='sum')
        ).assign(cancellation_rate=lambda x: x['cancelled_bookings'] / x['total_bookings'] * 100)
        top_countries = country_bookings.nlargest(10, 'total_bookings')
        
        if save_fig:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_countries.reset_index(), x='total_bookings', y='country')
            plt.title('Top 10 Countries by Number of Bookings', fontsize=14)
            plt.xlabel('Number of Bookings', fontsize=12)
            plt.ylabel('Country', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "top_countries_bookings")
            
            if self.plotly_enabled:
                fig = px.bar(top_countries.reset_index(), x='total_revenue', y='country', 
                            title='Top 10 Countries by Total Revenue', orientation='h')
                self._save_plot(fig, "top_countries_revenue", use_plotly=True)
            else:
                plt.figure(figsize=(12, 6))
                sns.barplot(data=top_countries.reset_index(), x='total_revenue', y='country')
                plt.title('Top 10 Countries by Total Revenue', fontsize=14)
                plt.xlabel('Total Revenue', fontsize=12)
                plt.ylabel('Country', fontsize=12)
                plt.grid(True, alpha=0.3)
                self._save_plot(plt.gcf(), "top_countries_revenue")
        
        result = {
            'country_bookings': country_bookings.reset_index().to_dict('records'),
            'top_countries': top_countries.reset_index().to_dict('records')
        }
        self.analytics_results['geographical_distribution'] = result
        return result
    
    def lead_time_analysis(self, save_fig: bool = True) -> Dict[str, Any]:
        lead_time_stats = self.df['lead_time'].describe().to_dict()
        lead_time_category = self.df.groupby('lead_time_category', observed=True).agg(
            count=pd.NamedAgg(column='booking_id', aggfunc='count'),
            percentage=pd.NamedAgg(column='booking_id', aggfunc=lambda x: len(x) / len(self.df) * 100),
            cancellation_rate=pd.NamedAgg(column='is_canceled', aggfunc=lambda x: x.mean() * 100)
        )
        lead_time_hotel = self.df.groupby('hotel')['lead_time'].agg(['mean', 'median']).reset_index()
        
        if save_fig:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df['lead_time'], bins=50, kde=True)
            plt.title('Lead Time Distribution', fontsize=14)
            plt.xlabel('Lead Time (days)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "lead_time_distribution")
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=lead_time_category.reset_index(), x='lead_time_category', y='percentage')
            plt.title('Lead Time Category Distribution', fontsize=14)
            plt.xlabel('Lead Time Category', fontsize=12)
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "lead_time_category_distribution")
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=lead_time_hotel, x='hotel', y='mean')
            plt.title('Mean Lead Time by Hotel Type', fontsize=14)
            plt.xlabel('Hotel Type', fontsize=12)
            plt.ylabel('Mean Lead Time (days)', fontsize=12)
            plt.grid(True, alpha=0.3)
            self._save_plot(plt.gcf(), "lead_time_by_hotel")
        
        result = {
            'lead_time_stats': lead_time_stats,
            'lead_time_category': lead_time_category.reset_index().to_dict('records'),
            'lead_time_hotel': lead_time_hotel.to_dict('records')
        }
        self.analytics_results['lead_time_analysis'] = result
        return result
    
    def generate_all_analytics(self, save_fig: bool = True) -> Dict[str, Any]:
        try:
            self.revenue_trends(save_fig)
            self.cancellation_analysis(save_fig)
            self.geographical_distribution(save_fig)
            self.lead_time_analysis(save_fig)
            with open(self.output_dir / "analytics_results.json", "w") as f:
                json.dump(self.analytics_results, f, cls=NumpyEncoder)
            logger.info("All analytics generated successfully")
            return self.analytics_results
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            raise

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

if __name__ == "__main__":
    analytics = HotelAnalytics(data_path="processed_hotel_bookings.csv.gz")
    results = analytics.generate_all_analytics()
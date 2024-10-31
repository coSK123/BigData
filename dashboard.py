import streamlit as st
import boto3
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
from decimal import Decimal
from dotenv import load_dotenv
import os
from typing import Dict, List, Tuple, Optional
import json
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logging() -> None:
    """Configure logging with both file and console handlers."""
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = RotatingFileHandler(
        'logs/dashboard.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class ElectionDashboard:
    def __init__(self):
        """Initialize the dashboard with AWS credentials and logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Election Dashboard")
        
        try:
            load_dotenv()
            session = boto3.Session(
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name='eu-north-1'
            )
            self.dynamodb = session.resource('dynamodb')
            self.cleansed_table = self.dynamodb.Table('Cleansed-Posts')
            self.cache_duration = 3600  # 1 hour cache
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {str(e)}", exc_info=True)
            raise

    def parse_date(self, date_str: str) -> datetime:
        """Parse ISO format date string from the database.
        
        Args:
            date_str: String in ISO format "2024-10-14T12:32:34.311422"
            
        Returns:
            datetime object
        """
        try:
            return datetime.fromisoformat(date_str)
        except ValueError as e:
            self.logger.error(f"Failed to parse date {date_str}: {str(e)}")
            raise

    @st.cache_data(ttl=3600, persist="disk")
    def get_date_range(_self) -> Tuple[datetime, datetime]:
        """Get min and max dates from the dataset."""
        try:
            _self.logger.info("Fetching date range from database")
            response = _self.cleansed_table.scan(
                ProjectionExpression='#date',
                ExpressionAttributeNames={'#date': 'Date'},
                Limit=1000
            )
            
            dates = [_self.parse_date(item['Date']) for item in response['Items']]
            
            if not dates:
                _self.logger.warning("No dates found in database")
                return datetime.now(), datetime.now()
            
            min_date, max_date = min(dates), max(dates)
            _self.logger.info(f"Date range: {min_date} to {max_date}")
            return min_date, max_date
            
        except Exception as e:
            _self.logger.error(f"Error getting date range: {str(e)}", exc_info=True)
            raise

    @st.cache_data(ttl=3600, persist="disk")
    def get_aggregated_metrics(_self, start_date: datetime, end_date: datetime) -> Dict:
        """Get pre-aggregated metrics for the dashboard."""
        try:
            _self.logger.info(f"Fetching metrics for date range: {start_date} to {end_date}")
            
            # Convert to ISO format strings for DynamoDB query
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            total_posts = 0
            total_engagement = 0
            unique_authors = set()
            languages = set()
            
            last_evaluated_key = None
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    },
                    'Limit': 100
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                
                response = _self.cleansed_table.scan(**scan_params)
                items = response['Items']
                
                for item in items:
                    total_posts += 1
                    metrics = item.get('metrics', {})
                    total_engagement += float(metrics.get('replies', 0))
                    total_engagement += float(metrics.get('retweets', 0))
                    total_engagement += float(metrics.get('favorites', 0))
                    
                    unique_authors.add(item.get('author', {}).get('id'))
                    languages.add(item.get('language'))
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            metrics_result = {
                'total_posts': total_posts,
                'total_engagement': total_engagement,
                'unique_authors': len(unique_authors),
                'languages': len(languages)
            }
            
            _self.logger.info(f"Retrieved metrics: {metrics_result}")
            return metrics_result
            
        except Exception as e:
            _self.logger.error(f"Error getting aggregated metrics: {str(e)}", exc_info=True)
            raise

    @st.cache_data(ttl=3600, persist="disk")
    def get_candidate_trends(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get daily candidate mentions with improved aggregation."""
        try:
            _self.logger.info(f"Fetching candidate trends for date range: {start_date} to {end_date}")
            
            # Convert to ISO format strings for DynamoDB query
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            mentions = defaultdict(lambda: defaultdict(int))
            
            last_evaluated_key = None
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    },
                    'Limit': 100
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                    
                response = _self.cleansed_table.scan(**scan_params)
                
                for item in response['Items']:
                    try:
                        date = _self.parse_date(item['Date']).date()
                        for candidate in item.get('candidates_mentioned', []):
                            mentions[date][candidate] += 1
                    except (ValueError, KeyError) as e:
                        _self.logger.warning(f"Skipping malformed item: {str(e)}")
                        continue
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            # Convert to DataFrame
            df_data = []
            for date, candidates in mentions.items():
                for candidate, count in candidates.items():
                    df_data.append({
                        'Date': pd.to_datetime(date),
                        'Candidate': candidate,
                        'Mentions': count
                    })
            
            if not df_data:
                _self.logger.warning("No candidate trends data found for the specified date range")
                return pd.DataFrame(columns=['Date', 'Candidate', 'Mentions', 'Mentions_MA'])
            
            df = pd.DataFrame(df_data)
            
            # Ensure we have all dates for all candidates
            date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
            candidates = df['Candidate'].unique()
            
            multi_idx = pd.MultiIndex.from_product([date_range, candidates], 
                                                names=['Date', 'Candidate'])
            
            df = df.set_index(['Date', 'Candidate']).reindex(multi_idx, fill_value=0)
            df = df.reset_index()
            
            # Calculate 3-day moving average
            df = df.sort_values(['Candidate', 'Date'])
            df['Mentions_MA'] = df.groupby('Candidate')['Mentions'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            _self.logger.info(f"Retrieved trends data with shape {df.shape}")
            return df
            
        except Exception as e:
            _self.logger.error(f"Error getting candidate trends: {str(e)}", exc_info=True)
            raise

def main():
    """Main function to run the Streamlit dashboard."""
    try:
        setup_logging()
        logger = logging.getLogger("main")
        logger.info("Starting dashboard application")
        
        st.set_page_config(layout="wide", page_title="Election Social Media Analysis")
        st.title("🗳️ US Election Social Media Analysis Dashboard")
        
        dashboard = ElectionDashboard()
        
        # Date range filter
        try:
            min_date, max_date = dashboard.get_date_range()
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(
                    min_date.date(),
                    max_date.date()
                )
            )
        except Exception as e:
            logger.error(f"Error setting up date range: {str(e)}")
            st.error("Failed to load date range. Please check the logs for details.")
            return
        
        if len(date_range) == 2:
            try:
                # Convert dates to datetime with time boundaries
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[1], datetime.max.time())
                
                # Get aggregated metrics
                metrics = dashboard.get_aggregated_metrics(start_date, end_date)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", metrics['total_posts'])
                with col2:
                    st.metric("Total Engagement", int(metrics['total_engagement']))
                with col3:
                    st.metric("Unique Authors", metrics['unique_authors'])
                with col4:
                    st.metric("Languages", metrics['languages'])
                
                # Display candidate trends
                st.header("📈 Candidate Mention Trends")
                trends_df = dashboard.get_candidate_trends(start_date, end_date)

                if not trends_df.empty and 'Date' in trends_df.columns:
                    fig = px.line(
                        trends_df,
                        x='Date',
                        y='Mentions_MA',
                        color='Candidate',
                        title="Daily Candidate Mentions (3-day moving average)",
                        labels={'Mentions_MA': 'Mentions', 'Date': 'Date'},
                    )
                    
                    fig.update_traces(
                        line=dict(width=2),
                        mode='lines+markers',
                        marker=dict(size=6)
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True
                        ),
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor='rgba(255,255,255,0.8)'
                        ),
                        hovermode='x unified'
                    )
                    
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_mentions = trends_df.groupby('Candidate')['Mentions'].mean().round(1)
                        st.markdown("**Average Daily Mentions**")
                        st.markdown(f"Biden: {avg_mentions.get('biden', 0)}")
                        st.markdown(f"Trump: {avg_mentions.get('trump', 0)}")
                        st.markdown(f"Harris: {avg_mentions.get('harris', 0)}")
                    
                    with col2:
                        total_mentions = trends_df.groupby('Candidate')['Mentions'].sum()
                        st.markdown("**Total Mentions**")
                        st.markdown(f"Biden: {total_mentions.get('biden', 0)}")
                        st.markdown(f"Trump: {total_mentions.get('trump', 0)}")
                        st.markdown(f"Harris: {total_mentions.get('harris', 0)}")
                    
                    with col3:
                        max_mentions = trends_df.groupby('Candidate')['Mentions'].max()
                        st.markdown("**Peak Daily Mentions**")
                        st.markdown(f"Biden: {max_mentions.get('biden', 0)}")
                        st.markdown(f"Trump: {max_mentions.get('trump', 0)}")
                        st.markdown(f"Harris: {max_mentions.get('harris', 0)}")
                else:
                    st.warning("No data available for the selected date range.")
                    
            except Exception as e:
                logger.error(f"Error processing data: {str(e)}", exc_info=True)
                st.error("""
                    An error occurred while processing the data. 
                    Please check the logs for details or try again later.
                """)
        else:
            st.info("Please select both start and end dates to view the analysis.")
            
    except Exception as e:
        logger.error(f"Critical dashboard error: {str(e)}", exc_info=True)
        st.error("""
            A critical error occurred while running the dashboard. 
            Please contact support with the following error details:
        """)
        st.code(str(e))

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them."""
    logger = logging.getLogger("error_handler")
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    # Display error in Streamlit
    st.error("""
        An unexpected error occurred. 
        The error has been logged and will be investigated.
        Please try again later.
    """)

if __name__ == "__main__":
    try:
        # Set up global exception handler
        import sys
        sys.excepthook = handle_uncaught_exception
        
        # Run the main application
        main()
        
    except KeyboardInterrupt:
        logger = logging.getLogger("main")
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger = logging.getLogger("main")
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
        st.error("""
            Failed to start the application. 
            Please check the logs for details or contact support.
        """)
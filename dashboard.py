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


def setup_logging() -> logging.Logger:
    """Configure logging with different levels for console and file"""
    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    
    # More aggressive filtering of AWS-related logs
    logging.getLogger('boto3').setLevel(logging.ERROR)
    logging.getLogger('botocore').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('s3transfer').setLevel(logging.ERROR)
    logging.getLogger('boto3.resources').setLevel(logging.ERROR)
    logging.getLogger('botocore.hooks').setLevel(logging.ERROR)
    logging.getLogger('botocore.auth').setLevel(logging.ERROR)
    logging.getLogger('botocore.regions').setLevel(logging.ERROR)
    logging.getLogger('botocore.parsers').setLevel(logging.ERROR)
    logging.getLogger('botocore.retryhandler').setLevel(logging.ERROR)
    logging.getLogger('botocore.endpoint').setLevel(logging.ERROR)
    logging.getLogger('botocore.httpsession').setLevel(logging.ERROR)
    
    # Detailed formatter for file logging
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Simplified formatter for console
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # File Handler - Detailed logging
    file_handler = RotatingFileHandler(
        'logs/dashboard.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create a stricter filter
    class StrictBotoFilter(logging.Filter):
        def filter(self, record):
            # Only allow logs from our app namespace
            return record.name.startswith('dashboard')

    file_handler.addFilter(StrictBotoFilter())
    
    # Console Handler - Only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(StrictBotoFilter())
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create and configure the application logger
    app_logger = logging.getLogger('dashboard')
    app_logger.setLevel(logging.DEBUG)
    
    return app_logger

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
            
            # Add cache clearing capability
            self.clear_cache()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {str(e)}", exc_info=True)
            raise
    def clear_cache(self):
        """Clear all Streamlit caches."""
        self.get_date_range.clear()
        self.get_aggregated_metrics.clear()
        self.get_candidate_trends.clear()
        self.get_candidate_sentiments.clear()
        self.get_engagement_metrics.clear()
        self.get_content_metrics.clear()
        self.get_geographical_metrics.clear()
        self.get_viral_metrics.clear()
        self.get_correlation_metrics.clear()
        self.get_comparative_metrics.clear()

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

    @st.cache_data(ttl=3600, persist="disk")
    def get_candidate_sentiments(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get daily candidate sentiment scores with aggregation."""
        try:
            _self.logger.info(f"Fetching candidate sentiments for date range: {start_date} to {end_date}")
            
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            # Dictionary to store daily sentiment scores
            sentiments = defaultdict(lambda: defaultdict(list))
            
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
                        sentiment_data = item.get('sentiments', {})
                        
                        # Collect all sentiment scores for each candidate
                        for entity, sentiment in sentiment_data.items():
                            if isinstance(sentiment, dict):  # Ensure we have a valid sentiment object
                                compound_score = float(sentiment.get('compound_score', 0))
                                sentiments[date][entity].append(compound_score)
                                
                    except (ValueError, KeyError) as e:
                        _self.logger.warning(f"Skipping malformed sentiment item: {str(e)}")
                        continue
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            # Convert to DataFrame with daily averages
            df_data = []
            for date, candidates in sentiments.items():
                for candidate, scores in candidates.items():
                    if scores:  # Only add if we have scores
                        avg_sentiment = sum(scores) / len(scores)
                        df_data.append({
                            'Date': pd.to_datetime(date),
                            'Candidate': candidate,
                            'Sentiment': avg_sentiment
                        })
            
            if not df_data:
                _self.logger.warning("No sentiment data found for the specified date range")
                return pd.DataFrame(columns=['Date', 'Candidate', 'Sentiment', 'Sentiment_MA'])
            
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
            df['Sentiment_MA'] = df.groupby('Candidate')['Sentiment'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            _self.logger.info(f"Retrieved sentiment data with shape {df.shape}")
            return df
            
        except Exception as e:
            _self.logger.error(f"Error getting candidate sentiments: {str(e)}", exc_info=True)
            raise


    @st.cache_data(ttl=3600, persist="disk")
    def get_engagement_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get engagement metrics with hourly breakdown."""
        try:
            _self.logger.info(f"Fetching engagement metrics for date range: {start_date} to {end_date}")
            
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            engagement_data = []
            last_evaluated_key = None
            
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    }
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                    
                response = _self.cleansed_table.scan(**scan_params)
                
                for item in response['Items']:
                    try:
                        date = _self.parse_date(item['Date'])
                        metrics = item.get('metrics', {})
                        
                        total_engagement = (
                            float(metrics.get('replies', 0)) +
                            float(metrics.get('retweets', 0)) +
                            float(metrics.get('favorites', 0))
                        )
                        
                        # Calculate engagement rate as total engagement divided by author followers
                        author = item.get('author', {})
                        follower_count = float(author.get('follower_count', 1))
                        engagement_rate = (total_engagement / follower_count) * 100 if follower_count > 0 else 0
                        
                        engagement_data.append({
                            'hour': date.hour,
                            'Candidate': next(iter(item.get('candidates_mentioned', [])), 'Unknown'),
                            'engagement_rate': engagement_rate,
                            'total_engagement': total_engagement,
                            'author_id': author.get('id'),
                            'author_name': author.get('id'),  # Using ID as name since name isn't stored
                            'follower_count': follower_count,
                            'post_count': 1
                        })
                        
                    except Exception as e:
                        _self.logger.warning(f"Error processing engagement item: {str(e)}")
                        continue
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return pd.DataFrame(engagement_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting engagement metrics: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, persist="disk")
    def get_content_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get content analysis metrics."""
        try:
            _self.logger.info(f"Fetching content metrics for date range: {start_date} to {end_date}")
            
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            content_data = []
            last_evaluated_key = None
            
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    }
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                    
                response = _self.cleansed_table.scan(**scan_params)
                
                for item in response['Items']:
                    content_data.append({
                        'source_device': item.get('source_device', 'Unknown'),
                        'language': item.get('language', 'Unknown'),
                        'count': 1
                    })
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return pd.DataFrame(content_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting content metrics: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, persist="disk")
    def get_geographical_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get geographical distribution metrics."""
        try:
            _self.logger.info(f"Fetching geographical metrics for date range: {start_date} to {end_date}")
            
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            geo_data = []
            last_evaluated_key = None
            
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    }
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                    
                response = _self.cleansed_table.scan(**scan_params)
                
                for item in response['Items']:
                    location = item.get('location', {})
                    if location and location.get('state'):
                        # Calculate average sentiment for the post
                        sentiments = item.get('sentiments', {})
                        sentiment_scores = [
                            float(s.get('compound_score', 0)) 
                            for s in sentiments.values()
                            if isinstance(s, dict)
                        ]
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                        
                        metrics = item.get('metrics', {})
                        total_engagement = sum(float(metrics.get(k, 0)) for k in ['replies', 'retweets', 'favorites'])
                        
                        geo_data.append({
                            'state': location['state'],
                            'sentiment_score': avg_sentiment,
                            'post_count': 1,
                            'engagement_rate': total_engagement
                        })
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return pd.DataFrame(geo_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting geographical metrics: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, persist="disk")
    def get_viral_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get viral content metrics."""
        try:
            _self.logger.info(f"Fetching viral metrics for date range: {start_date} to {end_date}")
            
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            viral_data = []
            last_evaluated_key = None
            
            while True:
                scan_params = {
                    'FilterExpression': '#date BETWEEN :start AND :end',
                    'ExpressionAttributeNames': {'#date': 'Date'},
                    'ExpressionAttributeValues': {
                        ':start': start_str,
                        ':end': end_str
                    }
                }
                
                if last_evaluated_key:
                    scan_params['ExclusiveStartKey'] = last_evaluated_key
                    
                response = _self.cleansed_table.scan(**scan_params)
                
                for item in response['Items']:
                    metrics = item.get('metrics', {})
                    total_engagement = sum(float(metrics.get(k, 0)) for k in ['replies', 'retweets', 'favorites'])
                    
                    # Get the first mentioned candidate or 'Unknown'
                    candidate = next(iter(item.get('candidates_mentioned', [])), 'Unknown')
                    
                    viral_data.append({
                        'text': item.get('text', ''),
                        'candidate': candidate,
                        'engagement_score': float(total_engagement),
                        'quality_score': float(item.get('quality_score', 0)),
                        'sentiment_score': float(
                            item.get('sentiments', {}).get(candidate, {}).get('compound_score', 0)
                        ),
                        'follower_count': float(item.get('author', {}).get('follower_count', 0)),
                        'virality_score': float(total_engagement) * float(item.get('quality_score', 0))
                    })
                
                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
            
            return pd.DataFrame(viral_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting viral metrics: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, persist="disk")
    def get_correlation_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get metrics for correlation analysis."""
        try:
            _self.logger.info(f"Fetching correlation metrics for date range: {start_date} to {end_date}")
            
            # Combine sentiment and engagement data
            sentiment_df = _self.get_candidate_sentiments(start_date, end_date)
            engagement_df = _self.get_engagement_metrics(start_date, end_date)
            
            # Merge and calculate additional metrics
            sentiment_df['date'] = pd.to_datetime(sentiment_df['Date']).dt.date
            engagement_df['date'] = pd.to_datetime(engagement_df['hour'], unit='h').dt.date
            
            corr_data = pd.merge(
                sentiment_df,
                engagement_df.groupby(['date', 'Candidate']).agg({
                    'engagement_rate': 'mean',
                    'total_engagement': 'sum'
                }).reset_index(),
                on=['date', 'Candidate'],
                how='outer'
            )
            
            # Calculate sentiment volatility
            corr_data['sentiment_volatility'] = corr_data.groupby('Candidate')['Sentiment'].transform(
                lambda x: x.rolling(window=3, min_periods=1).std()
            )
            
            return corr_data
            
        except Exception as e:
            _self.logger.error(f"Error getting correlation metrics: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600, persist="disk")
    def get_comparative_metrics(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get metrics for comparative analysis."""
        try:
            _self.logger.info(f"Fetching comparative metrics for date range: {start_date} to {end_date}")
            
            # Get all necessary metrics
            sentiment_df = _self.get_candidate_sentiments(start_date, end_date)
            engagement_df = _self.get_engagement_metrics(start_date, end_date)
            mentions_df = _self.get_candidate_trends(start_date, end_date)
            
            comparative_data = []
            
            # Process each candidate
            for candidate in sentiment_df['Candidate'].unique():
                metrics = {
                    'candidate': candidate,
                    'avg_sentiment': sentiment_df[sentiment_df['Candidate'] == candidate]['Sentiment'].mean(),
                    'sentiment_stability': 1 / (sentiment_df[sentiment_df['Candidate'] == candidate]['Sentiment'].std() + 0.1),
                    'total_mentions': mentions_df[mentions_df['Candidate'] == candidate]['Mentions'].sum(),
                    'avg_engagement': engagement_df[engagement_df['Candidate'] == candidate]['engagement_rate'].mean(),
                    'peak_mentions': mentions_df[mentions_df['Candidate'] == candidate]['Mentions'].max()
                }
                
                # Add each metric as a separate row
                for metric, value in metrics.items():
                    if metric != 'candidate':
                        comparative_data.append({
                            'candidate': candidate,
                            'metric': metric,
                            'value': value
                        })
            
            return pd.DataFrame(comparative_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting comparative metrics: {str(e)}")
            return pd.DataFrame()
        
def display_top_influencers(engagement_df: pd.DataFrame):
    """Display table of top influencers and their metrics."""
    try:
        # Group by author and calculate influence metrics
        influencers = engagement_df.groupby('author_id').agg({
            'author_name': 'first',
            'follower_count': 'first',
            'engagement_score': 'mean',
            'post_count': 'size',
            'total_engagement': 'sum',
            'verified': 'first'
        }).reset_index()
        
        # Calculate influence score
        influencers['influence_score'] = (
            influencers['engagement_score'] * 
            np.log1p(influencers['follower_count']) * 
            influencers['post_count']
        )
        
        # Sort and get top 10
        top_influencers = influencers.nlargest(10, 'influence_score')
        
        # Create formatted table
        st.dataframe(
            top_influencers[[
                'author_name', 'follower_count', 'post_count',
                'total_engagement', 'influence_score', 'verified'
            ]].style
            .format({
                'follower_count': '{:,.0f}',
                'total_engagement': '{:,.0f}',
                'influence_score': '{:,.2f}'
            })
            .background_gradient(subset=['influence_score'])
        )
        
    except Exception as e:
        st.error("Failed to display top influencers")
        raise e

def display_state_sentiment_table(geo_df: pd.DataFrame):
    """Display table of sentiment analysis by state."""
    try:
        # Calculate average sentiment and post volume by state
        state_metrics = geo_df.groupby('state').agg({
            'sentiment_score': ['mean', 'std'],
            'post_count': 'sum',
            'engagement_rate': 'mean'
        }).round(3)
        
        # Reset column names
        state_metrics.columns = [
            'Avg Sentiment', 'Sentiment Std', 
            'Total Posts', 'Avg Engagement'
        ]
        
        # Sort by post volume
        state_metrics = state_metrics.sort_values('Total Posts', ascending=False)
        
        # Display as formatted table
        st.dataframe(
            state_metrics.style
            .format({
                'Avg Sentiment': '{:+.3f}',
                'Sentiment Std': '{:.3f}',
                'Total Posts': '{:,.0f}',
                'Avg Engagement': '{:.1%}'
            })
            .background_gradient(subset=['Avg Sentiment'], cmap='RdYlGn')
            .background_gradient(subset=['Total Posts'], cmap='Purples')
        )
        
    except Exception as e:
        st.error("Failed to display state sentiment table")
        raise e

def display_viral_posts_table(viral_df: pd.DataFrame):
    """Display table of top viral posts with metrics."""
    try:
        # Sort posts by virality score
        top_posts = viral_df.nlargest(10, 'virality_score')
        
        # Create formatted table
        st.dataframe(
            top_posts[[
                'text', 'candidate', 'engagement_score',
                'quality_score', 'sentiment_score', 'virality_score'
            ]].style
            .format({
                'engagement_score': '{:.2f}',
                'quality_score': '{:.2f}',
                'sentiment_score': '{:+.2f}',
                'virality_score': '{:.2f}'
            })
            .background_gradient(subset=['virality_score'])
            .set_properties(**{'text-align': 'left'})
        )
        
    except Exception as e:
        st.error("Failed to display viral posts table")
        raise e

def display_sentiment_flow(corr_df: pd.DataFrame):
    """Display sentiment flow patterns and transitions."""
    try:
        # Create columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment transitions over time
            fig = px.line(
                corr_df,
                x='date',
                y='sentiment_ma',
                color='candidate',
                title="Sentiment Moving Average",
                labels={'sentiment_ma': 'Sentiment MA', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment stability
            fig = px.box(
                corr_df,
                x='candidate',
                y='sentiment_volatility',
                title="Sentiment Volatility by Candidate"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display top sentiment swing periods
        st.subheader("Notable Sentiment Swings")
        sentiment_swings = calculate_sentiment_swings(corr_df)
        st.dataframe(
            sentiment_swings.style
            .format({
                'swing_magnitude': '{:+.2f}',
                'duration_hours': '{:.1f}'
            })
            .background_gradient(subset=['swing_magnitude'], cmap='RdYlGn')
        )
        
    except Exception as e:
        st.error("Failed to display sentiment flow")
        raise e

def display_comparative_table(comp_df: pd.DataFrame):
    """Display detailed comparative metrics table."""
    try:
        # Pivot the comparison metrics
        comparison_table = comp_df.pivot(
            index='metric',
            columns='candidate',
            values='value'
        ).round(3)
        
        # Calculate relative differences
        candidates = comparison_table.columns
        for c1 in candidates:
            for c2 in candidates:
                if c1 < c2:  # Avoid duplicate comparisons
                    comparison_table[f'{c1} vs {c2}'] = (
                        comparison_table[c1] / comparison_table[c2] - 1
                    )
        
        # Display formatted table
        st.dataframe(
            comparison_table.style
            .format({
                col: '{:,.2f}' for col in candidates
            })
            .format({
                col: '{:+.1%}' for col in comparison_table.columns 
                if ' vs ' in col
            })
            .background_gradient(
                subset=[col for col in comparison_table.columns if ' vs ' in col],
                cmap='RdYlGn'
            )
        )
        
    except Exception as e:
        st.error("Failed to display comparative table")
        raise e

def calculate_sentiment_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate significant sentiment changes over time."""
    try:
        swings = []
        
        for candidate in df['candidate'].unique():
            candidate_data = df[df['candidate'] == candidate].copy()
            
            # Calculate sentiment changes
            candidate_data['sentiment_change'] = candidate_data['sentiment_ma'].diff()
            
            # Find significant swings (e.g., > 2 standard deviations)
            threshold = candidate_data['sentiment_change'].std() * 2
            significant_changes = candidate_data[
                abs(candidate_data['sentiment_change']) > threshold
            ]
            
            for _, row in significant_changes.iterrows():
                swings.append({
                    'candidate': candidate,
                    'start_date': row['date'] - pd.Timedelta(hours=24),
                    'end_date': row['date'],
                    'swing_magnitude': row['sentiment_change'],
                    'duration_hours': 24,
                    'final_sentiment': row['sentiment_ma']
                })
        
        return pd.DataFrame(swings).sort_values('swing_magnitude', ascending=False)
        
    except Exception as e:
        st.error("Failed to calculate sentiment swings")
        raise e

def format_post_text(text: str, max_length: int = 100) -> str:
    """Format post text for display in tables."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def create_engagement_tooltip(row: pd.Series) -> str:
    """Create detailed tooltip for engagement metrics."""
    return (
        f"Total Engagement: {row['total_engagement']:,.0f}\n"
        f"Engagement Rate: {row['engagement_rate']:.1%}\n"
        f"Quality Score: {row['quality_score']:.2f}"
    )

def display_engagement_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display advanced engagement analysis."""
    try:
        st.header("💫 Engagement Analysis")
        
        engagement_df = dashboard.get_engagement_metrics(start_date, end_date)
        if not engagement_df.empty:
            # Engagement by time of day
            st.subheader("Engagement Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    engagement_df,
                    x='hour',
                    color='Candidate',
                    title="Activity by Hour of Day",
                    labels={'hour': 'Hour', 'count': 'Number of Posts'},
                    nbins=24
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    engagement_df,
                    x='Candidate',
                    y='engagement_rate',
                    title="Engagement Rate Distribution",
                    labels={'engagement_rate': 'Engagement Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Display top influencers
            st.subheader("Top Influencers")
            display_top_influencers(engagement_df)

    except Exception as e:
        st.error("Failed to load engagement analysis")
        raise e

def display_content_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display content pattern analysis."""
    try:
        st.header("📝 Content Analysis")
        
        content_df = dashboard.get_content_metrics(start_date, end_date)
        if not content_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Source Distribution")
                fig = px.pie(
                    content_df, 
                    names='source_device',
                    values='count',
                    title="Posts by Source"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Language Distribution")
                fig = px.bar(
                    content_df.groupby('language').size().reset_index(name='count'),
                    x='language',
                    y='count',
                    title="Posts by Language"
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Failed to load content analysis")
        raise e

def display_geographical_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display geographical distribution of posts."""
    try:
        st.header("🗺️ Geographical Analysis")
        
        geo_df = dashboard.get_geographical_metrics(start_date, end_date)
        if not geo_df.empty:
            # State-wise distribution
            fig = px.choropleth(
                geo_df,
                locations='state',
                locationmode="USA-states",
                color='post_count',
                scope="usa",
                title="Posts by State",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # State-wise sentiment
            st.subheader("Sentiment by State")
            display_state_sentiment_table(geo_df)

    except Exception as e:
        st.error("Failed to load geographical analysis")
        raise e

def display_virality_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display analysis of viral content and patterns."""
    try:
        st.header("🌟 Viral Content Analysis")
        
        viral_df = dashboard.get_viral_metrics(start_date, end_date)
        if not viral_df.empty:
            # Viral posts metrics
            st.subheader("Top Viral Posts")
            display_viral_posts_table(viral_df)
            
            # Virality patterns
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    viral_df,
                    x='follower_count',
                    y='engagement_score',
                    color='Candidate',
                    title="Engagement vs Follower Count",
                    log_x=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    viral_df,
                    x='Candidate',
                    y='quality_score',
                    title="Content Quality Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Failed to load virality analysis")
        raise e

def display_sentiment_correlation(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display correlations between sentiment and other metrics."""
    try:
        st.header("🔄 Sentiment Correlation Analysis")
        
        corr_df = dashboard.get_correlation_metrics(start_date, end_date)
        if not corr_df.empty:
            # Correlation heatmap
            fig = px.imshow(
                corr_df.corr(),
                title="Metric Correlations",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment flow analysis
            st.subheader("Sentiment Flow Patterns")
            display_sentiment_flow(corr_df)

    except Exception as e:
        st.error("Failed to load correlation analysis")
        raise e

def display_comparative_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display comparative analysis between candidates."""
    try:
        st.header("⚖️ Comparative Analysis")
        
        comp_df = dashboard.get_comparative_metrics(start_date, end_date)
        if not comp_df.empty:
            # Radar chart of metrics
            fig = px.line_polar(
                comp_df,
                r='value',
                theta='metric',
                color='Candidate',
                line_close=True,
                title="Candidate Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Comparison")
            display_comparative_table(comp_df)

    except Exception as e:
        st.error("Failed to load comparative analysis")
        raise e

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


def display_overview_metrics(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display the overview metrics section."""
    try:
        # Get aggregated metrics
        metrics = dashboard.get_aggregated_metrics(start_date, end_date)
        
        # Create metrics section
        st.header("📊 Overview Metrics")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Posts", 
                f"{metrics['total_posts']:,}",
                help="Total number of posts analyzed in selected period"
            )
        with col2:
            st.metric(
                "Total Engagement", 
                f"{int(metrics['total_engagement']):,}",
                help="Sum of replies, retweets, and favorites"
            )
        with col3:
            st.metric(
                "Unique Authors", 
                f"{metrics['unique_authors']:,}",
                help="Number of unique users posting content"
            )
        with col4:
            st.metric(
                "Languages", 
                metrics['languages'],
                help="Number of different languages detected"
            )
            
    except Exception as e:
        st.error("Failed to load overview metrics")
        raise e

def display_mentions_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display the mentions trend analysis section."""
    try:
        st.header("📈 Candidate Mention Trends")
        
        # Get mentions data
        trends_df = dashboard.get_candidate_trends(start_date, end_date)

        if not trends_df.empty and 'Date' in trends_df.columns:
            # Create mentions trend chart
            fig = create_mentions_chart(trends_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display mentions statistics
            st.subheader("Mention Statistics")
            display_mentions_statistics(trends_df)
        else:
            st.warning("No mention data available for the selected date range.")
            
    except Exception as e:
        st.error("Failed to load mentions analysis")
        raise e

def display_sentiment_analysis(dashboard: ElectionDashboard, start_date: datetime, end_date: datetime):
    """Display the sentiment analysis section."""
    try:
        st.header("🎭 Candidate Sentiment Trends")
        
        # Get sentiment data
        sentiment_df = dashboard.get_candidate_sentiments(start_date, end_date)

        if not sentiment_df.empty and 'Date' in sentiment_df.columns:
            # Create sentiment trend chart
            fig = create_sentiment_chart(sentiment_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sentiment statistics
            st.subheader("Sentiment Statistics")
            display_sentiment_statistics(sentiment_df)
        else:
            st.warning("No sentiment data available for the selected date range.")
            
    except Exception as e:
        st.error("Failed to load sentiment analysis")
        raise e

def create_mentions_chart(trends_df: pd.DataFrame) -> go.Figure:
    """Create the mentions trend chart."""
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
    return fig

def create_sentiment_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """Create the sentiment trend chart."""
    fig = px.line(
        sentiment_df,
        x='Date',
        y='Sentiment_MA',
        color='Candidate',
        title="Daily Candidate Sentiment Scores (3-day moving average)",
        labels={'Sentiment_MA': 'Average Sentiment', 'Date': 'Date'},
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
            showgrid=True,
            range=[-1, 1]
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
    
    # Add a zero line to show neutral sentiment
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def display_mentions_statistics(trends_df: pd.DataFrame):
    """Display the mentions statistics in columns."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_mentions = trends_df.groupby('Candidate')['Mentions'].mean().round(1)
        st.markdown("**Average Daily Mentions**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {avg_mentions.get(candidate, 0):,.1f}")
    
    with col2:
        total_mentions = trends_df.groupby('Candidate')['Mentions'].sum()
        st.markdown("**Total Mentions**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {total_mentions.get(candidate, 0):,.0f}")
    
    with col3:
        max_mentions = trends_df.groupby('Candidate')['Mentions'].max()
        st.markdown("**Peak Daily Mentions**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {max_mentions.get(candidate, 0):,.0f}")

def display_sentiment_statistics(sentiment_df: pd.DataFrame):
    """Display the sentiment statistics in columns."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sentiment = sentiment_df.groupby('Candidate')['Sentiment'].mean().round(3)
        st.markdown("**Average Sentiment**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {avg_sentiment.get(candidate, 0):.3f}")
    
    with col2:
        max_sentiment = sentiment_df.groupby('Candidate')['Sentiment'].max().round(3)
        st.markdown("**Most Positive Day**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {max_sentiment.get(candidate, 0):.3f}")
    
    with col3:
        min_sentiment = sentiment_df.groupby('Candidate')['Sentiment'].min().round(3)
        st.markdown("**Most Negative Day**")
        for candidate in ['biden', 'trump', 'harris']:
            st.markdown(f"{candidate.title()}: {min_sentiment.get(candidate, 0):.3f}")

def main():
    """Main function to run the Streamlit dashboard."""
    try:
        # Setup and initialization
        logger = setup_logging()
        logger.info("Starting dashboard application")
        
        # Configure Streamlit page
        st.set_page_config(
            layout="wide", 
            page_title="Election Social Media Analysis",
            page_icon="🗳️"
        )
        st.title("🗳️ US Election Social Media Analysis Dashboard")
        
        # Initialize dashboard
        dashboard = ElectionDashboard()

        # Add debug mode in sidebar
        st.sidebar.header("Dashboard Controls")
        debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
        
        if debug_mode:
            if st.sidebar.button("Clear Cache"):
                dashboard.clear_cache()
                st.success("Cache cleared successfully!")
        
        # Date range selection with debug info
        try:
            min_date, max_date = dashboard.get_date_range()
            if debug_mode:
                st.sidebar.write(f"Min date: {min_date}")
                st.sidebar.write(f"Max date: {max_date}")
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        except Exception as e:
            logger.error(f"Error setting up date range: {str(e)}")
            st.error("Failed to load date range. Please check the logs for details.")
            if debug_mode:
                st.error(f"Date range error: {str(e)}")
            return

        # Main dashboard content with debug info
        if len(date_range) == 2:
            try:
                # Convert dates to datetime with time boundaries
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[1], datetime.max.time())
                
                if debug_mode:
                    st.sidebar.write("Selected date range:")
                    st.sidebar.write(f"Start: {start_date}")
                    st.sidebar.write(f"End: {end_date}")

                # Create tabs
                tabs = st.tabs([
                    "Basic Metrics",
                    "Engagement",
                    "Content",
                    "Geography",
                    "Virality",
                    "Correlations",
                    "Comparison"
                ])
                
                # Display data loading status in debug mode
                if debug_mode:
                    status = st.sidebar.empty()
                
                # Load and display each tab with debug info
                with tabs[0]:
                    if debug_mode:
                        status.write("Loading Basic Metrics...")
                    display_overview_metrics(dashboard, start_date, end_date)
                    display_mentions_analysis(dashboard, start_date, end_date)
                    display_sentiment_analysis(dashboard, start_date, end_date)
                
                with tabs[1]:
                    if debug_mode:
                        status.write("Loading Engagement Analysis...")
                    engagement_df = dashboard.get_engagement_metrics(start_date, end_date)
                    if debug_mode and not engagement_df.empty:
                        st.sidebar.write(f"Engagement data shape: {engagement_df.shape}")
                    try:
                        display_engagement_analysis(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in engagement analysis: {str(e)}")
                        st.error("Failed to display engagement analysis")
                
                with tabs[2]:
                    if debug_mode:
                        status.write("Loading Content Analysis...")
                    content_df = dashboard.get_content_metrics(start_date, end_date)
                    if debug_mode and not content_df.empty:
                        st.sidebar.write(f"Content data shape: {content_df.shape}")
                    try:
                        display_content_analysis(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in content analysis: {str(e)}")
                        st.error("Failed to display content analysis")
                
                with tabs[3]:
                    if debug_mode:
                        status.write("Loading Geographical Analysis...")
                    geo_df = dashboard.get_geographical_metrics(start_date, end_date)
                    if debug_mode and not geo_df.empty:
                        st.sidebar.write(f"Geographical data shape: {geo_df.shape}")
                    try:
                        display_geographical_analysis(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in geographical analysis: {str(e)}")
                        st.error("Failed to display geographical analysis")
                
                with tabs[4]:
                    if debug_mode:
                        status.write("Loading Virality Analysis...")
                    viral_df = dashboard.get_viral_metrics(start_date, end_date)
                    if debug_mode and not viral_df.empty:
                        st.sidebar.write(f"Viral data shape: {viral_df.shape}")
                    try:
                        display_virality_analysis(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in virality analysis: {str(e)}")
                        st.error("Failed to display virality analysis")
                
                with tabs[5]:
                    if debug_mode:
                        status.write("Loading Correlation Analysis...")
                    corr_df = dashboard.get_correlation_metrics(start_date, end_date)
                    if debug_mode and not corr_df.empty:
                        st.sidebar.write(f"Correlation data shape: {corr_df.shape}")
                    try:
                        display_sentiment_correlation(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in correlation analysis: {str(e)}")
                        st.error("Failed to display correlation analysis")
                
                with tabs[6]:
                    if debug_mode:
                        status.write("Loading Comparative Analysis...")
                    comp_df = dashboard.get_comparative_metrics(start_date, end_date)
                    if debug_mode and not comp_df.empty:
                        st.sidebar.write(f"Comparative data shape: {comp_df.shape}")
                    try:
                        display_comparative_analysis(dashboard, start_date, end_date)
                    except Exception as e:
                        logger.error(f"Error in comparative analysis: {str(e)}")
                        st.error("Failed to display comparative analysis")
                
                if debug_mode:
                    status.write("All data loaded successfully!")
                
            except Exception as e:
                logger.error(f"Error processing dashboard data: {str(e)}", exc_info=True)
                st.error("An error occurred while processing the data.")
                if debug_mode:
                    st.error(f"Processing error: {str(e)}")
                
        else:
            st.info("Please select both start and end dates to view the analysis.")

    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
        st.error(
            "Failed to start the application. "
            "Please check the logs for details or contact support."
        )
        if debug_mode:
            st.error(f"Critical error: {str(e)}")

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
        st.error(
            "Failed to start the application. "
            "Please check the logs for details or contact support."
        )
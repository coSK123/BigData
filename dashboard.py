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
from typing import Dict, List
import json

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class ElectionDashboard:
    def __init__(self):
        load_dotenv()
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-north-1'
        )
        self.dynamodb = session.resource('dynamodb')
        self.cleansed_table = self.dynamodb.Table('Cleansed-Posts')
        
        # Configure cache for aggregated data
        self.cache_duration = 3600  # 1 hour cache

    def decimal_to_float(self, obj):
        """Convert Decimal objects to float recursively"""
        if isinstance(obj, list):
            return [self.decimal_to_float(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.decimal_to_float(value) for key, value in obj.items()}
        elif isinstance(obj, Decimal):
            return float(obj)
        return obj

    @st.cache_data(ttl=3600, persist="disk")
    def get_date_range(_self):
        """Get min and max dates from the dataset"""
        response = _self.cleansed_table.scan(
            ProjectionExpression='#date',
            ExpressionAttributeNames={'#date': 'Date'},
            Limit=1000
        )
        dates = [datetime.strptime(item['Date'], '%a %b %d %H:%M:%S %z %Y') for item in response['Items']]
        return min(dates), max(dates)

    @st.cache_data(ttl=3600, persist="disk")
    def get_aggregated_metrics(_self, start_date: str, end_date: str) -> Dict:
        """Get pre-aggregated metrics for the dashboard"""
        total_posts = 0
        total_engagement = 0
        unique_authors = set()
        languages = set()
        
        # Process in batches
        last_evaluated_key = None
        while True:
            if last_evaluated_key:
                response = _self.cleansed_table.scan(
                    FilterExpression='#date BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#date': 'Date'},
                    ExpressionAttributeValues={
                        ':start': start_date,
                        ':end': end_date
                    },
                    ExclusiveStartKey=last_evaluated_key,
                    Limit=100
                )
            else:
                response = _self.cleansed_table.scan(
                    FilterExpression='#date BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#date': 'Date'},
                    ExpressionAttributeValues={
                        ':start': start_date,
                        ':end': end_date
                    },
                    Limit=100
                )
            
            items = response['Items']
            
            # Process batch
            for item in items:
                total_posts += 1
                metrics = item.get('metrics', {})
                total_engagement += float(metrics.get('replies', 0))
                total_engagement += float(metrics.get('retweets', 0))
                total_engagement += float(metrics.get('favorites', 0))
                
                unique_authors.add(item.get('author', {}).get('id'))
                languages.add(item.get('language'))
            
            # Check if more data to process
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        return {
            'total_posts': total_posts,
            'total_engagement': total_engagement,
            'unique_authors': len(unique_authors),
            'languages': len(languages)
        }

    @st.cache_data(ttl=3600, persist="disk")
    def get_candidate_trends(_self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get daily candidate mentions with improved aggregation"""
        mentions = defaultdict(lambda: defaultdict(int))
        
        last_evaluated_key = None
        while True:
            scan_params = {
                'FilterExpression': '#date BETWEEN :start AND :end',
                'ExpressionAttributeNames': {'#date': 'Date'},
                'ExpressionAttributeValues': {
                    ':start': start_date,
                    ':end': end_date
                },
                'Limit': 100
            }
            
            if last_evaluated_key:
                scan_params['ExclusiveStartKey'] = last_evaluated_key
                
            response = _self.cleansed_table.scan(**scan_params)
            
            for item in response['Items']:
                # Extract just the date part
                date = datetime.strptime(item['Date'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d')
                for candidate in item['candidates_mentioned']:
                    mentions[date][candidate] += 1
            
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
        
        # Check if we have any data
        if not df_data:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Date', 'Candidate', 'Mentions', 'Mentions_MA'])
        
        df = pd.DataFrame(df_data)
        
        # Ensure we have all dates for all candidates (fill missing with 0)
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        candidates = df['Candidate'].unique()
        
        # Create multi-index with all combinations
        multi_idx = pd.MultiIndex.from_product([date_range, candidates], 
                                            names=['Date', 'Candidate'])
        
        # Reindex and fill missing values
        df = df.set_index(['Date', 'Candidate']).reindex(multi_idx, fill_value=0)
        df = df.reset_index()
        
        # Calculate 3-day moving average to smooth the lines
        df = df.sort_values(['Candidate', 'Date'])
        df['Mentions_MA'] = df.groupby('Candidate')['Mentions'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return df

def main():
    st.set_page_config(layout="wide", page_title="Election Social Media Analysis")
    
    st.title("🗳️ US Election Social Media Analysis Dashboard")
    
    dashboard = ElectionDashboard()
    
    # Date range filter
    min_date, max_date = dashboard.get_date_range()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(
            min_date.date(),
            max_date.date()
        )
    )
    # Check if both dates are selected to prevent errors 
    if len(date_range) == 2:

        # Get aggregated metrics
        metrics = dashboard.get_aggregated_metrics(
            date_range[0].isoformat(),
            date_range[1].isoformat()
        )
        
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
        trends_df = dashboard.get_candidate_trends(
            date_range[0].isoformat(),
            date_range[1].isoformat()
        )

        if not trends_df.empty:
            # Create the plot with improved styling
            fig = px.line(
                trends_df,
                x='Date',
                y='Mentions_MA',  # Use moving average
                color='Candidate',
                title="Daily Candidate Mentions (3-day moving average)",
                labels={'Mentions_MA': 'Mentions', 'Date': 'Date'},
            )
            
            # Customize the plot
            fig.update_traces(
                line=dict(width=2),  # Make lines thicker
                mode='lines+markers',  # Add markers at data points
                marker=dict(size=6)
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
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
                hovermode='x unified'  # Show all values for a given x-coordinate
            )
            
            # Add range selector
            fig.update_xaxes(rangeslider_visible=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_mentions = trends_df.groupby('Candidate')['Mentions'].mean().round(1)
                st.markdown("**Average Daily Mentions**", unsafe_allow_html=True)
                st.markdown(f"<small>Biden: {avg_mentions.get('biden', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Trump: {avg_mentions.get('trump', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Harris: {avg_mentions.get('harris', 0)}</small>", unsafe_allow_html=True)
            
            with col2:
                total_mentions = trends_df.groupby('Candidate')['Mentions'].sum()
                st.markdown("**Total Mentions**", unsafe_allow_html=True)
                st.markdown(f"<small>Biden: {total_mentions.get('biden', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Trump: {total_mentions.get('trump', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Harris: {total_mentions.get('harris', 0)}</small>", unsafe_allow_html=True)
            
            with col3:
                max_mentions = trends_df.groupby('Candidate')['Mentions'].max()
                st.markdown("**Peak Daily Mentions**", unsafe_allow_html=True)
                st.markdown(f"<small>Biden: {max_mentions.get('biden', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Trump: {max_mentions.get('trump', 0)}</small>", unsafe_allow_html=True)
                st.markdown(f"<small>Harris: {max_mentions.get('harris', 0)}</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
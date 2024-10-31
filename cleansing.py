from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import re

import os
from dotenv import load_dotenv



from models import TwitterPost, ProcessedFlag
import boto3
from datetime import datetime
import simplejson

import re
from typing import Optional, Dict, Any

from decimal import Decimal

class DataCleansingService:
    def __init__(self, raw_table: str, cleansed_table: str, batch_size: int = 100):
        
        load_dotenv()

        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-north-1'
        )
        self.dynamodb = session.resource('dynamodb')
        self.raw_table = self.dynamodb.Table(raw_table)
        self.cleansed_table = self.dynamodb.Table(cleansed_table)
        self.batch_size = batch_size
        
        # Keywords for candidate and party identification
        self.candidates = {
            'trump': ['trump', 'donald trump', 'donald j trump'],
            'harris': ['harris', 'kamala', 'kamala harris', 'vice president harris'],
            'biden': ['biden', 'joe biden', 'president biden']
        }
        self.parties = {
            'republican': ['republican', 'gop', 'rnc'],
            'democrat': ['democrat', 'democratic', 'dnc']
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\u263a-\U0001f645]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()

    def extract_mentions(self, text: str) -> Dict[str, list]:
        """Extract candidate and party mentions"""
        text = text.lower()
        mentions = {
            'candidates': [],
            'parties': []
        }
        
        for candidate, keywords in self.candidates.items():
            if any(keyword in text for keyword in keywords):
                mentions['candidates'].append(candidate)
                
        for party, keywords in self.parties.items():
            if any(keyword in text for keyword in keywords):
                mentions['parties'].append(party)
                
        return mentions

    def extract_location(self, location: Optional[str]) -> Optional[Dict[str, str]]:
        """Clean and structure location data"""
        if not location:
            return None
            
        # Extract state from location if possible
        us_state_pattern = r'\b([A-Z]{2})\b'
        state_match = re.search(us_state_pattern, location.upper())
        
        return {
            'raw_location': location,
            'state': state_match.group(1) if state_match else None
        }

    def cleanse_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform raw post into cleansed format"""
        try:
            if not all(k in post for k in ['text', 'Date', 'ID']):
                return None

            text = self.clean_text(post['text'])
            if not text:
                return None

            mentions = self.extract_mentions(text)
            if not mentions['candidates'] and not mentions['parties']:
                return None

            user = post.get('user', {})
            
            # Convert Decimal metrics to integers
            metrics = {
                'replies': int(post.get('reply_count', 0)) if isinstance(post.get('reply_count'), (Decimal, int)) else 0,
                'retweets': int(post.get('retweet_count', 0)) if isinstance(post.get('retweet_count'), (Decimal, int)) else 0,
                'favorites': int(post.get('favorite_count', 0)) if isinstance(post.get('favorite_count'), (Decimal, int)) else 0,
                'views': int(post.get('views', 0)) if isinstance(post.get('views'), (Decimal, int)) else 0
            }
            
            cleansed_post = {
                'ID': post['ID'],  # Primary key
                'Date': post['Date'],  # Sort key
                'text': text,
                'created_at': post['Date'],
                'processed_at': datetime.now().isoformat(),
                'candidates_mentioned': mentions['candidates'],
                'parties_mentioned': mentions['parties'],
                'metrics': metrics,
                'author': {
                    'id': user.get('user_id'),
                    'verified': bool(user.get('is_verified', False)),
                    'follower_count': int(user.get('follower_count', 0)) if isinstance(user.get('follower_count'), (Decimal, int)) else 0
                },
                'location': self.extract_location(user.get('location')),
                'is_retweet': bool(post.get('retweet', False)),
                'language': post.get('language'),
                'source_device': post.get('source')
            }
            
            return cleansed_post
            
        except Exception as e:
            print(f"Error processing post {post.get('ID')}: {str(e)}")
            return None

    def process_batch(self):
        """Process batch of unprocessed raw posts"""
        try:
            response = self.raw_table.scan(
                FilterExpression='attribute_not_exists(processed_flag) OR processed_flag = :val',
                ExpressionAttributeValues={':val': ProcessedFlag.NOT_PROCESSED.value},
                Limit=self.batch_size
            )
            raw_posts = response['Items']
            
            for post in raw_posts:
                cleansed_post = self.cleanse_post(post)
                if cleansed_post:
                    self.cleansed_table.put_item(Item=cleansed_post)
                    
                    self.raw_table.update_item(
                        Key={'ID': post['ID']},
                        UpdateExpression='SET processed_flag = :val, processed_at = :time',
                        ExpressionAttributeValues={
                            ':val': ProcessedFlag.PROCESSED.value,
                            ':time': datetime.now().isoformat()
                        }
                    )
                else:
                    self.raw_table.update_item(
                        Key={'ID': post['ID']},
                        UpdateExpression='SET processed_flag = :val, processed_at = :time',
                        ExpressionAttributeValues={
                            ':val': ProcessedFlag.ERROR.value,
                            ':time': datetime.now().isoformat()
                        }
                    )
        except Exception as e:
            print(f"Batch processing error: {str(e)}")

    def remaining_posts(self):
        """Get count of remaining unprocessed posts"""
        response = self.raw_table.scan(
            FilterExpression='attribute_not_exists(processed_flag) OR processed_flag = :val',
            ExpressionAttributeValues={':val': ProcessedFlag.NOT_PROCESSED.value}
        )
        return response['Count']

    def first_post_to_json(self):
        response = self.raw_table.scan(Limit=1)
        post = response['Items'][0]
        with open('first_post.json', 'w') as f:
            simplejson.dump(post, f, indent=4, use_decimal=True)


    def process_deamon(self):
        """Daemon process with error handling and proper sleep import"""
        from time import sleep
        while self.remaining_posts() > 0:
            try:
                self.process_batch()
                print(f"Processed Batch. Remaining posts to process: {self.remaining_posts()}")
                sleep(10)
            except Exception as e:
                print(f"Daemon error: {str(e)}")
                sleep(30)  # Longer sleep on error

    # background processing through threading
    def start(self):
        import threading
        thread = threading.Thread(target=self.process_deamon)
        thread.start() 





if __name__ == '__main__':
    service = DataCleansingService(raw_table='Raw-Posts', cleansed_table='Cleansed-Posts', batch_size=1)
    service.start()



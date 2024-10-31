from datetime import datetime
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
import logging
import os
from dotenv import load_dotenv
from enum import Enum
import time
import simplejson
from typing import List, Optional, Dict, Any
from models import ProcessedFlag
from decimal import Decimal
import asyncio
import re
import random
from pydantic import BaseModel

class ProcessedFlag(Enum):
    NOT_PROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2
    ERROR = -1

class PostMetrics(BaseModel):
    replies: int = 0
    retweets: int = 0
    favorites: int = 0
    views: int = 0

class Author(BaseModel):
    id: Optional[str]
    verified: bool = False
    follower_count: int = 0

class Location(BaseModel):
    raw_location: str
    state: Optional[str]

class CleansedPost(BaseModel):
    ID: str
    Date: str
    text: str
    created_at: str
    processed_at: str
    candidates_mentioned: List[str]
    parties_mentioned: List[str]
    metrics: PostMetrics
    author: Author
    location: Optional[Location]
    is_retweet: bool = False
    language: Optional[str]
    source_device: Optional[str]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleansingService:
    def __init__(self, raw_table: str, cleansed_table: str, batch_size: int = 25):
        load_dotenv()
        
        self.session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-north-1'
        )
        self.dynamodb = self.session.resource('dynamodb')
        self.raw_table = self.dynamodb.Table(raw_table)
        self.cleansed_table = self.dynamodb.Table(cleansed_table)
        # Reduced batch size to prevent throughput issues
        self.batch_size = min(batch_size, 25)
        
        # Initialize exponential backoff parameters
        self.base_delay = 1  # Base delay in seconds
        self.max_retries = 5
        self.max_delay = 32  # Maximum delay in seconds
        
        self._initialize_reference_data()

    def _initialize_reference_data(self):
        """Initialize reference data for filtering"""
        self.candidates = {
            'trump': ['trump', 'donald trump', 'donald j trump'],
            'harris': ['harris', 'kamala', 'kamala harris', 'vice president harris'],
            'biden': ['biden', 'joe biden', 'president biden']
        }
        self.parties = {
            'republican': ['republican', 'gop', 'rnc'],
            'democrat': ['democrat', 'democratic', 'dnc']
        }

    async def _exponential_backoff(self, attempt: int):
        """Implement exponential backoff with jitter"""
        if attempt > 0:
            delay = min(self.max_delay, self.base_delay * (2 ** attempt))
            jitter = delay * 0.1 * random.random()  # Add 10% jitter
            await asyncio.sleep(delay + jitter)

    async def _scan_with_backoff(self, **kwargs):
        """Perform DynamoDB scan with backoff and pagination handling"""
        for attempt in range(self.max_retries):
            try:
                items = []
                last_evaluated_key = None
                
                while True:
                    if last_evaluated_key:
                        kwargs['ExclusiveStartKey'] = last_evaluated_key
                        
                    response = self.raw_table.scan(**kwargs)
                    
                    if 'Items' in response:
                        items.extend(response['Items'])
                    
                    last_evaluated_key = response.get('LastEvaluatedKey')
                    
                    # If we're just counting or we've hit our limit, we can stop
                    if 'Select' in kwargs or (len(items) >= kwargs.get('Limit', float('inf'))):
                        break
                        
                    # If no more pages, stop
                    if not last_evaluated_key:
                        break
                
                # If we're counting, return the count response
                if 'Select' in kwargs and kwargs['Select'] == 'COUNT':
                    return response
                
                # Otherwise return items with pagination structure
                return {
                    'Items': items[:kwargs.get('Limit', len(items))],
                    'Count': len(items),
                    'LastEvaluatedKey': last_evaluated_key
                }
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

    async def _update_with_backoff(self, key: dict, update_expr: str, expr_values: dict):
        """Perform DynamoDB update with backoff strategy"""
        for attempt in range(self.max_retries):
            try:
                return self.raw_table.update_item(
                    Key=key,
                    UpdateExpression=update_expr,
                    ExpressionAttributeValues=expr_values
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

    async def _put_item_with_backoff(self, table, item):
        """Perform DynamoDB put_item with backoff strategy"""
        for attempt in range(self.max_retries):
            try:
                return table.put_item(Item=item)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Put throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

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


    async def remaining_posts(self):
        """Get count of remaining unprocessed posts"""
        try:
            response = await self._scan_with_backoff(
                FilterExpression=Attr('processed_flag').not_exists() | 
                               Attr('processed_flag').eq(ProcessedFlag.NOT_PROCESSED.value),
                Select='COUNT'
            )
            return response.get('Count', 0)
        except Exception as e:
            logger.error(f"Error counting remaining posts: {str(e)}")
            return 0

    async def process_batch(self):
        """Process batch of unprocessed raw posts with improved logging"""
        try:
            logger.info("Starting batch processing...")
            scan_result = await self._scan_with_backoff(
                FilterExpression=Attr('processed_flag').not_exists() | 
                            Attr('processed_flag').eq(ProcessedFlag.NOT_PROCESSED.value),
                Limit=self.batch_size
            )
            
            raw_posts = scan_result.get('Items', [])
            logger.info(f"Found {len(raw_posts)} unprocessed posts in current batch")
            
            if not raw_posts:
                remaining = await self.remaining_posts()
                logger.info(f"No posts in current batch. Total remaining: {remaining}")
                return 0

            processed_count = 0
            for post in raw_posts:
                try:
                    logger.debug(f"Processing post {post.get('ID')}")
                    # Mark as processing
                    await self._update_with_backoff(
                        key={'ID': post['ID'], 'Date': post['Date']},
                        update_expr='SET processed_flag = :val, processing_started = :time',
                        expr_values={
                            ':val': ProcessedFlag.PROCESSING.value,
                            ':time': datetime.now().isoformat()
                        }
                    )

                    cleansed_post = self.cleanse_post(post)
                    if cleansed_post:
                        await self._put_item_with_backoff(self.cleansed_table, cleansed_post)
                        
                        await self._update_with_backoff(
                            key={'ID': post['ID'], 'Date': post['Date']},
                            update_expr='SET processed_flag = :val, processed_at = :time',
                            expr_values={
                                ':val': ProcessedFlag.PROCESSED.value,
                                ':time': datetime.now().isoformat()
                            }
                        )
                        processed_count += 1
                        logger.debug(f"Successfully processed post {post.get('ID')}")
                    else:
                        await self._update_with_backoff(
                            key={'ID': post['ID'], 'Date': post['Date']},
                            update_expr='SET processed_flag = :val, processed_at = :time',
                            expr_values={
                                ':val': ProcessedFlag.ERROR.value,
                                ':time': datetime.now().isoformat()
                            }
                        )
                        logger.warning(f"Post {post.get('ID')} failed cleansing")

                    # Add delay between posts due to low capacity
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing post {post.get('ID')}: {str(e)}")
                    continue

            logger.info(f"Batch complete. Processed {processed_count} posts successfully")
            return processed_count

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return 0

    async def process_daemon(self):
        """Daemon process with improved logging and error handling"""
        while True:
            try:
                remaining = await self.remaining_posts()
                logger.info(f"Starting new processing cycle. Total remaining posts: {remaining}")
                
                if remaining == 0:
                    logger.info("No more posts to process")
                    break
                    
                processed = await self.process_batch()
                logger.info(f"Cycle complete. Processed {processed} posts. Remaining: {remaining - processed}")
                
                # Longer sleeps due to low capacity
                if processed == 0:
                    logger.info("No posts processed in this cycle. Sleeping for 30 seconds...")
                    await asyncio.sleep(30)
                else:
                    logger.info(f"Sleeping for 10 seconds before next cycle...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Daemon error: {str(e)}")
                logger.info("Sleeping for 60 seconds due to error...")
                await asyncio.sleep(60)

    def start(self):
        """Start the processing daemon"""
        asyncio.run(self.process_daemon())

if __name__ == '__main__':
    service = DataCleansingService(
        raw_table='Raw-Posts',
        cleansed_table='Cleansed-Posts',
        batch_size=10  # Very small batch size due to low capacity
    )
    service.start()



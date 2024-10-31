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
from pydantic import BaseModel, Field, field_validator
from logging.handlers import RotatingFileHandler

class ProcessedFlag(Enum):
    NOT_PROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2
    ERROR = -1

class PostMetrics(BaseModel):
    replies: int = Field(default=0, ge=0)
    retweets: int = Field(default=0, ge=0)
    favorites: int = Field(default=0, ge=0)
    views: int = Field(default=0, ge=0)

class Author(BaseModel):
    id: Optional[str] = None
    verified: bool = False
    follower_count: int = Field(default=0, ge=0)

class Location(BaseModel):
    raw_location: str
    state: Optional[str] = None

class RawPost(BaseModel):
    ID: str
    Date: str
    text: str
    reply_count: Optional[int] = Field(default=0, ge=0)
    retweet_count: Optional[int] = Field(default=0, ge=0)
    favorite_count: Optional[int] = Field(default=0, ge=0)
    views: Optional[int] = Field(default=0, ge=0)
    user: Optional[Dict[str, Any]] = None
    retweet: bool = False
    language: Optional[str] = None
    source: Optional[str] = None

    @field_validator('Date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            # Parse the Twitter-style date format
            dt = datetime.strptime(v, '%a %b %d %H:%M:%S %z %Y')
            # Convert to ISO format with microseconds
            return dt.astimezone().isoformat()
        except ValueError as e:
            raise ValueError(f'Invalid date format: {e}')

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
    language: Optional[str] = None
    source_device: Optional[str] = None

    def to_dynamodb_dict(self) -> dict:
        """Convert the model to a DynamoDB-compatible dictionary"""
        data = self.model_dump(exclude_none=True)
        
        # Convert nested models to dictionaries
        if 'metrics' in data:
            data['metrics'] = self.metrics.model_dump(exclude_none=True)
        if 'author' in data:
            data['author'] = self.author.model_dump(exclude_none=True)
        if 'location' in data and self.location:
            data['location'] = self.location.model_dump(exclude_none=True)
            
        return data

def setup_logging() -> None:
    """
    Configure logging with different levels for console and file:
    - Console: Only INFO and above, minimal format
    - File: DEBUG and above, detailed format with all information
    """
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Detailed formatter for file logging
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Simplified formatter for console
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File Handler - Detailed logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = RotatingFileHandler(
        'logs/cleansing.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console Handler - Only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only WARNING and above
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create a separate logger for the application
    app_logger = logging.getLogger('cleansing')
    app_logger.setLevel(logging.DEBUG)

    # Modify messaging in the DataCleansingService class
    return app_logger

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
        self.batch_size = min(batch_size, 25)
        
        self.base_delay = 1
        self.max_retries = 5
        self.max_delay = 32
        
        # Setup logging with new configuration
        self.logger = setup_logging()
        
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

    async def _scan_with_backoff(self, **kwargs) -> Dict[str, Any]:
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
                    
                    if 'Select' in kwargs or (len(items) >= kwargs.get('Limit', float('inf'))):
                        break
                        
                    if not last_evaluated_key:
                        break
                
                if 'Select' in kwargs and kwargs['Select'] == 'COUNT':
                    return response
                
                return {
                    'Items': items[:kwargs.get('Limit', len(items))],
                    'Count': len(items),
                    'LastEvaluatedKey': last_evaluated_key
                }
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    logging.warning(f"Throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
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
                    self.logger.warning(f"Throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

    async def _put_item_with_backoff(self, table, item: CleansedPost):
        """Perform DynamoDB put_item with backoff strategy"""
        for attempt in range(self.max_retries):
            try:
                # Convert Pydantic model to DynamoDB-compatible dictionary
                item_dict = item.to_dynamodb_dict()
                return table.put_item(Item=item_dict)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Put throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s\u263a-\U0001f645]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()

    def extract_mentions(self, text: str) -> Dict[str, List[str]]:
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

    def extract_location(self, location: Optional[str]) -> Optional[Location]:
        """Clean and structure location data"""
        if not location:
            return None
            
        us_state_pattern = r'\b([A-Z]{2})\b'
        state_match = re.search(us_state_pattern, location.upper())
        
        return Location(
            raw_location=location,
            state=state_match.group(1) if state_match else None
        )

    def cleanse_post(self, post: Dict[str, Any]) -> Optional[CleansedPost]:
        """Transform raw post into cleansed format"""
        try:
            raw_post = RawPost(**post)
            
            text = self.clean_text(raw_post.text)
            if not text:
                return None

            mentions = self.extract_mentions(text)
            if not mentions['candidates'] and not mentions['parties']:
                return None

            user = raw_post.user or {}
            
            metrics = PostMetrics(
                replies=int(raw_post.reply_count or 0),
                retweets=int(raw_post.retweet_count or 0),
                favorites=int(raw_post.favorite_count or 0),
                views=int(raw_post.views or 0)
            )
            
            author = Author(
                id=user.get('user_id'),
                verified=bool(user.get('is_verified', False)),
                follower_count=int(user.get('follower_count', 0))
            )
            
            cleansed = CleansedPost(
                ID=raw_post.ID,
                Date=raw_post.Date,
                text=text,
                created_at=raw_post.Date,
                processed_at=datetime.now().isoformat(),
                candidates_mentioned=mentions['candidates'],
                parties_mentioned=mentions['parties'],
                metrics=metrics,
                author=author,
                location=self.extract_location(user.get('location')),
                is_retweet=raw_post.retweet,
                language=raw_post.language,
                source_device=raw_post.source
            )
            
            return cleansed
            
        except Exception as e:
            self.logger.error(f"Error processing post {post.get('ID')}: {str(e)}")
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
            self.logger.error(f"Error counting remaining posts: {str(e)}")
            return 0

    async def process_batch(self):
        """Process batch of unprocessed raw posts with improved logging"""
        try:
            self.logger.debug("Starting batch processing...")  # Debug level for detailed logging
            scan_result = await self._scan_with_backoff(
                FilterExpression=Attr('processed_flag').not_exists() | 
                            Attr('processed_flag').eq(ProcessedFlag.NOT_PROCESSED.value),
                Limit=self.batch_size
            )
            
            raw_posts = scan_result.get('Items', [])
            self.logger.debug(f"Found {len(raw_posts)} unprocessed posts in current batch")
            
            if not raw_posts:
                remaining = await self.remaining_posts()
                self.logger.info(f"No posts in current batch. Total remaining: {remaining}")
                return 0

            processed_count = 0
            for post in raw_posts:
                try:
                    self.logger.debug(f"Processing post {post.get('ID')}")
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
                        self.logger.debug(f"Successfully processed post {post.get('ID')}")
                    else:
                        await self._update_with_backoff(
                            key={'ID': post['ID'], 'Date': post['Date']},
                            update_expr='SET processed_flag = :val, processed_at = :time',
                            expr_values={
                                ':val': ProcessedFlag.ERROR.value,
                                ':time': datetime.now().isoformat()
                            }
                        )
                        self.logger.warning(f"Post {post.get('ID')} failed cleansing")

                except Exception as e:
                    self.logger.error(f"Error processing post {post.get('ID')}: {str(e)}")
                    continue

            # Only log to console if there's an issue or significant progress
            if processed_count == 0:
                self.logger.warning(f"Batch complete but no posts were processed successfully")
            else:
                self.logger.info(f"Batch complete. Processed {processed_count} posts successfully")
            return processed_count

        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            return 0

    async def process_daemon(self):
        """Daemon process with improved logging"""
        while True:
            try:
                remaining = await self.remaining_posts()
                self.logger.info(f"Remaining posts to process: {remaining}")
                
                if remaining == 0:
                    self.logger.warning("No more posts to process - shutting down")
                    break
                    
                processed = await self.process_batch()
                
                # Only log to console if there's an issue
                if processed == 0:
                    self.logger.warning("No posts processed in this cycle. Sleeping for 30 seconds...")
                    await asyncio.sleep(30)
                else:
                    self.logger.debug(f"Sleeping for 10 seconds before next cycle...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                self.logger.error(f"Daemon error: {str(e)}")
                self.logger.warning("Sleeping for 60 seconds due to error...")
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



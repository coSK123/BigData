from datetime import datetime
import os
import random
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
import logging
from enum import Enum
import time
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
import asyncio
import hashlib
import json
import spacy
from transformers import pipeline
import re
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from decimal import Decimal

class ProcessedFlag(Enum):
    NOT_PROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2
    ERROR = -1

class EntitySentiment(BaseModel):
    """Sentiment analysis for a specific entity (candidate/party)"""
    entity: str
    sentiment: str  # positive, negative, neutral
    confidence: Decimal
    context: str
    compound_score: Decimal
    engagement_score: Decimal = Decimal('0.0')

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v)
        }


class PostMetrics(BaseModel):
    """Engagement metrics for a post"""
    replies: int = Field(default=0, ge=0)
    retweets: int = Field(default=0, ge=0)
    favorites: int = Field(default=0, ge=0)
    views: int = Field(default=0, ge=0)

class Author(BaseModel):
    """Author information"""
    id: Optional[str] = None
    verified: bool = False
    follower_count: int = Field(default=0, ge=0)

class Location(BaseModel):
    """Location information with optional state"""
    raw_location: str
    state: Optional[str] = None

class RawPost(BaseModel):
    """Model for raw posts from the collector"""
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
            dt = datetime.strptime(v, '%a %b %d %H:%M:%S %z %Y')
            return dt.astimezone().isoformat()
        except ValueError as e:
            raise ValueError(f'Invalid date format: {e}')

class CleansedPost(BaseModel):
    """Enhanced model for cleansed posts with sentiment analysis"""
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
    cleansing_version: str
    quality_score: Decimal
    sentiments: Dict[str, EntitySentiment]
    processing_metadata: Dict[str, Any]

    def to_dynamodb_dict(self) -> dict:
        """Convert the model to a DynamoDB-compatible dictionary"""
        data = self.model_dump(exclude_none=True)
        
        # Convert all float values to Decimal
        def convert_floats(obj):
            if isinstance(obj, float):
                return Decimal(str(obj))
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            return obj
        
        data = convert_floats(data)
        
        # Convert nested models to dictionaries
        if 'metrics' in data:
            data['metrics'] = convert_floats(self.metrics.model_dump(exclude_none=True))
        if 'author' in data:
            data['author'] = convert_floats(self.author.model_dump(exclude_none=True))
        if 'location' in data and self.location:
            data['location'] = convert_floats(self.location.model_dump(exclude_none=True))
        if 'sentiments' in data:
            data['sentiments'] = {
                k: convert_floats(v.model_dump(exclude_none=True))
                for k, v in self.sentiments.items()
            }
            
        return data

@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring cleansing process"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_fields: Dict[str, int] = field(default_factory=dict)
    malformed_dates: int = 0
    unknown_languages: int = 0
    empty_texts: int = 0
    processing_time: Decimal = field(default_factory=lambda: Decimal('0.0'))
    sentiment_failures: int = 0
    
    @property
    def validity_rate(self) -> Decimal:
        return Decimal(str((self.valid_records / self.total_records * 100))) if self.total_records > 0 else Decimal('0')
    
    def to_dict(self) -> Dict:
        # Convert all float values to Decimal
        def convert_to_decimal(val):
            if isinstance(val, float):
                return Decimal(str(val))
            elif isinstance(val, dict):
                return {k: convert_to_decimal(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert_to_decimal(i) for i in val]
            return val

        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'invalid_records': self.invalid_records,
            'missing_fields': self.missing_fields,
            'malformed_dates': self.malformed_dates,
            'unknown_languages': self.unknown_languages,
            'empty_texts': self.empty_texts,
            'validity_rate': self.validity_rate,
            'processing_time': self.processing_time,
            'sentiment_failures': self.sentiment_failures
        }
        
        return convert_to_decimal(metrics_dict)

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis for political posts using publicly available models"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load more comprehensive models
        self.nlp = spacy.load("en_core_web_lg")  # Larger model for better entity recognition
        
        # Use reliable, publicly available models
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        
        # Expanded reference data
        self.candidate_aliases = {
            'trump': [
                'trump', 'donald trump', 'donald j trump', 'former president trump',
                'president trump', '45th president', 'maga', 'trump2024'
            ],
            'harris': [
                'harris', 'kamala', 'kamala harris', 'vice president harris',
                'vp harris', 'madam vice president', 'kamala2024'
            ],
            'biden': [
                'biden', 'joe biden', 'president biden', 'potus',
                'joe', 'biden2024', 'bidenharris'
            ]
        }
        
        self.party_references = {
            'republican': [
                'republican', 'gop', 'rnc', 'conservative', 'right wing',
                'trumpist', 'red state', 'maga republican'
            ],
            'democrat': [
                'democrat', 'democratic', 'dnc', 'liberal', 'left wing',
                'progressive', 'blue state', 'biden democrat'
            ]
        }

        # Keywords for context analysis
        self.sentiment_keywords = {
            'positive': [
                'support', 'great', 'good', 'best', 'strong', 'excellent',
                'love', 'amazing', 'wonderful', 'success', 'win', 'victory'
            ],
            'negative': [
                'bad', 'worse', 'worst', 'weak', 'poor', 'terrible',
                'hate', 'awful', 'failure', 'lose', 'lost', 'corrupt'
            ]
        }

    def _normalize_entity(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Normalize entity to candidate and/or party"""
        text = text.lower()
        
        # Check candidates
        for candidate, aliases in self.candidate_aliases.items():
            if text in aliases:
                return candidate, None
                
        # Check parties
        for party, references in self.party_references.items():
            if text in references:
                return None, party
                
        return None, None

    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment score based on keyword presence"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] 
                           if word in words)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] 
                           if word in words)
        
        if positive_count == 0 and negative_count == 0:
            return 0
            
        return (positive_count - negative_count) / (positive_count + negative_count)

    def _calculate_engagement_score(self, text: str) -> float:
        """Calculate engagement score based on text features"""
        doc = self.nlp(text)
        
        # Factors that indicate high engagement
        exclamation_count = text.count('!')
        question_count = text.count('?')
        hashtag_count = len([t for t in doc if t.text.startswith('#')])
        mention_count = len([t for t in doc if t.text.startswith('@')])
        caps_words = len([t for t in doc if t.text.isupper() and len(t.text) > 1])
        
        # Normalize and combine scores
        return min(1.0, (
            exclamation_count * 0.2 +
            question_count * 0.15 +
            hashtag_count * 0.1 +
            mention_count * 0.05 +
            caps_words * 0.1
        ))

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Enhanced analysis combining model predictions with rule-based analysis"""
        try:
            doc = self.nlp(text)
            results = {}
            
            for sent in doc.sents:
                sent_text = str(sent)
                
                # Get base sentiment from model
                model_sentiment = self.sentiment_model(sent_text)[0]
                keyword_score = self._calculate_keyword_sentiment(sent_text)
                engagement_score = self._calculate_engagement_score(sent_text)
                
                for ent in sent.ents:
                    if ent.label_ in ["PERSON", "ORG", "NORP"]:
                        candidate, party = self._normalize_entity(ent.text)
                        entity = candidate or party
                        
                        if not entity:
                            continue
                        
                        # Convert model score to -1 to 1 range
                        model_score = Decimal(str(model_sentiment['score']))
                        if model_sentiment['label'] == 'LABEL_0':  # negative
                            model_score = -model_score
                        elif model_sentiment['label'] == 'LABEL_1':  # neutral
                            model_score = Decimal('0')
                        
                        # Combine model and keyword sentiment
                        keyword_contribution = Decimal(str(keyword_score))
                        compound_score = (
                            model_score * Decimal('0.7') +
                            keyword_contribution * Decimal('0.3')
                        )
                        
                        
                        result = EntitySentiment(
                            entity=entity,
                            sentiment=(
                                'positive' if compound_score > 0.1
                                else 'negative' if compound_score < -0.1
                                else 'neutral'
                            ),
                            engagement_score=engagement_score,
                            confidence=model_sentiment['score'],
                            context=sent_text,
                            compound_score=compound_score
                        )

                        # Keep strongest sentiment for each entity
                        if (
                            entity not in results or 
                            abs(compound_score) > abs(results[entity]['compound_score'])
                        ):
                            results[entity] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced sentiment analysis failed: {str(e)}", exc_info=True)
            return {}
        
class SentimentAnalyzer:
    """Handles sentiment analysis for political posts"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        
        # Initialize reference data
        self.candidate_aliases = {
            'trump': ['trump', 'donald trump', 'donald j trump'],
            'harris': ['harris', 'kamala', 'kamala harris', 'vice president harris'],
            'biden': ['biden', 'joe biden', 'president biden']
        }
        
        self.party_references = {
            'republican': ['republican', 'gop', 'rnc'],
            'democrat': ['democrat', 'democratic', 'dnc']
        }

    def _normalize_entity(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Normalize entity to candidate and/or party"""
        text = text.lower()
        
        # Check candidates
        for candidate, aliases in self.candidate_aliases.items():
            if text in aliases:
                return candidate, None
                
        # Check parties
        for party, references in self.party_references.items():
            if text in references:
                return None, party
                
        return None, None

    def _extract_context(self, text: str, start_idx: int, end_idx: int, window: int = 50) -> str:
        """Extract context around an entity mention"""
        start = max(0, start_idx - window)
        end = min(len(text), end_idx + window)
        return text[start:end]

    def analyze_text(self, text: str) -> Dict[str, EntitySentiment]:
        """Analyze sentiment for all political entities in text"""
        try:
            doc = self.nlp(text)
            results = {}
            
            for sent in doc.sents:
                for ent in sent.ents:
                    if ent.label_ in ["PERSON", "ORG"]:
                        candidate, party = self._normalize_entity(ent.text)
                        entity = candidate or party
                        
                        if not entity:
                            continue
                            
                        context = str(sent) if len(str(sent)) < 100 else self._extract_context(
                            text, ent.start_char, ent.end_char
                        )
                        
                        sentiment_result = self.sentiment_model(context)[0]
                        
                        sentiment = {
                            'LABEL_0': 'negative',
                            'LABEL_1': 'neutral',
                            'LABEL_2': 'positive'
                        }.get(sentiment_result['label'], 'neutral')
                        
                        # Convert scores to Decimal
                        score = Decimal(str(sentiment_result['score']))
                        compound_score = (
                            score if sentiment == 'positive'
                            else -score if sentiment == 'negative'
                            else Decimal('0')
                        )
                        
                        if (
                            entity not in results or 
                            abs(compound_score) > abs(results[entity].compound_score)
                        ):
                            results[entity] = EntitySentiment(
                                entity=entity,
                                sentiment=sentiment,
                                confidence=score,
                                context=context,
                                compound_score=compound_score
                            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return {}
        


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
        'logs/cleansing.log',
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
            return record.name.startswith('cleansing')

    file_handler.addFilter(StrictBotoFilter())
    
    # Console Handler - Only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(StrictBotoFilter())
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create and configure the application logger
    app_logger = logging.getLogger('cleansing')
    app_logger.setLevel(logging.DEBUG)
    
    return app_logger

class EnhancedCleansingService:
    """Enhanced service for cleansing social media posts with sentiment analysis"""
    
    def __init__(self, raw_table: str, cleansed_table: str, metrics_table: str, batch_size: int = 25):
        """Initialize the cleansing service"""
        load_dotenv()
        
        self.session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-north-1'
        )
        self.dynamodb = self.session.resource('dynamodb')
        self.raw_table = self.dynamodb.Table(raw_table)
        self.cleansed_table = self.dynamodb.Table(cleansed_table)
        self.metrics_table = self.dynamodb.Table(metrics_table)
        
        self.batch_size = min(batch_size, 25)
        self.base_delay = 1
        self.max_retries = 5
        self.max_delay = 32
        
        # Initialize components
        self.logger = setup_logging()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quality_metrics = DataQualityMetrics()
        
        # Generate version for this processing run
        self.version_id = self._generate_version_id()
        
    def _generate_version_id(self) -> str:
        """Generate a version identifier for this processing run"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.sha256(
            json.dumps({
                'batch_size': self.batch_size,
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment',
                'spacy_model': 'en_core_web_sm',
                'version': '2.0.0'  # Update this when making significant changes
            }).encode()
        ).hexdigest()[:8]
        
        return f"v{timestamp}_{config_hash}"

    async def _exponential_backoff(self, attempt: int):
        """Implement exponential backoff with jitter"""
        if attempt > 0:
            delay = min(self.max_delay, self.base_delay * (2 ** attempt))
            jitter = delay * 0.1 * random.random()
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
                    
                    if 'Select' in kwargs and kwargs['Select'] == 'COUNT':
                        return response
                        
                    if len(items) >= kwargs.get('Limit', float('inf')):
                        break
                        
                    if not last_evaluated_key:
                        break
                
                return {
                    'Items': items[:kwargs.get('Limit', len(items))],
                    'Count': len(items),
                    'LastEvaluatedKey': last_evaluated_key
                }
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
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

    async def _put_item_with_backoff(self, table, item: dict):
        """Perform DynamoDB put_item with backoff strategy"""
        for attempt in range(self.max_retries):
            try:
                return table.put_item(Item=item)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Put throughput exceeded, attempt {attempt + 1}/{self.max_retries}")
                    await self._exponential_backoff(attempt)
                else:
                    raise

    def calculate_quality_score(self, post: Dict[str, Any]) -> Decimal:
        """Calculate quality score for a post based on various factors"""
        score = Decimal('1.0')
        
        required_fields = ['text', 'Date', 'ID']
        for field in required_fields:
            if not post.get(field):
                score *= Decimal('0.7')
                self.quality_metrics.missing_fields[field] = \
                    self.quality_metrics.missing_fields.get(field, 0) + 1
        
        text = post.get('text', '')
        if not text:
            score *= Decimal('0.5')
            self.quality_metrics.empty_texts += 1
        elif len(text) < 10:
            score *= Decimal('0.8')
        
        if not post.get('user'):
            score *= Decimal('0.9')
        
        metrics = ['reply_count', 'retweet_count', 'favorite_count']
        for metric in metrics:
            if metric not in post:
                score *= Decimal('0.95')
        
        return max(Decimal('0.1'), score)

    async def cleanse_post(self, raw_post: Dict[str, Any]) -> Optional[CleansedPost]:
        """Clean and enhance a single post"""
        start_time = time.time()
        self.quality_metrics.total_records += 1
        
        try:
            # Parse raw post
            post = RawPost(**raw_post)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(raw_post)
            
            # Clean text
            cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', post.text)
            cleaned_text = re.sub(r'[^\w\s\u263a-\U0001f645]', ' ', cleaned_text)
            cleaned_text = ' '.join(cleaned_text.split())
            
            if not cleaned_text:
                self.quality_metrics.invalid_records += 1
                return None
            
            # Analyze sentiment
            try:
                sentiments = self.sentiment_analyzer.analyze_text(cleaned_text)
            except Exception as e:
                self.logger.error(f"Sentiment analysis failed: {str(e)}")
                self.quality_metrics.sentiment_failures += 1
                sentiments = {}
            
            # Extract mentioned candidates and parties from sentiment analysis
            candidates_mentioned = [
                entity for entity, sentiment in sentiments.items()
                if entity in self.sentiment_analyzer.candidate_aliases
            ]
            parties_mentioned = [
                entity for entity, sentiment in sentiments.items()
                if entity in self.sentiment_analyzer.party_references
            ]
            
            # Create metrics
            metrics = PostMetrics(
                replies=int(post.reply_count or 0),
                retweets=int(post.retweet_count or 0),
                favorites=int(post.favorite_count or 0),
                views=int(post.views or 0)
            )
            
            # Create author info
            user = post.user or {}
            author = Author(
                id=user.get('user_id'),
                verified=bool(user.get('is_verified', False)),
                follower_count=int(user.get('follower_count', 0))
            )
            
            # Create location info
            location = None
            if user.get('location'):
                state_match = re.search(r'\b([A-Z]{2})\b', user['location'].upper())
                location = Location(
                    raw_location=user['location'],
                    state=state_match.group(1) if state_match else None
                )
            
            # Create cleansed post
            cleansed = CleansedPost(
                ID=post.ID,
                Date=post.Date,
                text=cleaned_text,
                created_at=post.Date,
                processed_at=datetime.now().isoformat(),
                candidates_mentioned=candidates_mentioned,
                parties_mentioned=parties_mentioned,
                metrics=metrics,
                author=author,
                location=location,
                is_retweet=post.retweet,
                language=post.language,
                source_device=post.source,
                cleansing_version=self.version_id,
                quality_score=quality_score,
                sentiments=sentiments,
                processing_metadata={
                    'processing_time': time.time() - start_time,
                    'sentiment_success': bool(sentiments),
                    'quality_score': quality_score
                }
            )
            
            self.quality_metrics.valid_records += 1
            return cleansed
            
        except ValueError as e:
            self.logger.error(f"Validation error processing post: {str(e)}")
            self.quality_metrics.malformed_dates += 1
            return None
        except Exception as e:
            self.logger.error(f"Error processing post: {str(e)}")
            self.quality_metrics.invalid_records += 1
            return None
        finally:
            self.quality_metrics.processing_time += Decimal(str(time.time() - start_time))

    async def store_metrics(self):
        """Store current quality metrics"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'version': self.version_id,
                **self.quality_metrics.to_dict()
            }
            
            # Additional conversion for any floating point numbers in processing metadata
            def convert_floats_to_decimal(d):
                if isinstance(d, float):
                    return Decimal(str(d))
                elif isinstance(d, dict):
                    return {k: convert_floats_to_decimal(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [convert_floats_to_decimal(i) for i in d]
                return d
            
            metrics_data = convert_floats_to_decimal(metrics_data)
            
            await self._put_item_with_backoff(
                self.metrics_table,
                metrics_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {str(e)}")

    async def remaining_posts(self) -> int:
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

    async def process_batch(self) -> int:
        """Process a batch of posts with fixed scanning and error handling"""
        try:
            # Scan for unprocessed posts
            scan_result = await self._scan_with_backoff(
                FilterExpression=Attr('processed_flag').not_exists() | 
                                Attr('processed_flag').eq(ProcessedFlag.NOT_PROCESSED.value),
                Limit=self.batch_size
            )
            
            raw_posts = scan_result.get('Items', [])
            self.logger.debug(f"Found {len(raw_posts)} unprocessed posts")
            
            if not raw_posts:
                remaining = await self.remaining_posts()
                self.logger.info(f"No posts in current batch. Total remaining: {remaining}")
                return 0

            processed_count = 0
            
            for post in raw_posts:
                try:
                    # Mark as processing
                    await self._update_with_backoff(
                        key={'ID': post['ID'], 'Date': post['Date']},
                        update_expr='SET processed_flag = :val, processing_started = :time',
                        expr_values={
                            ':val': ProcessedFlag.PROCESSING.value,
                            ':time': datetime.now().isoformat()
                        }
                    )
                    
                    # Process post
                    cleansed_post = await self.cleanse_post(post)
                    
                    if cleansed_post:
                        # Store cleansed post
                        await self._put_item_with_backoff(
                            self.cleansed_table,
                            cleansed_post.to_dynamodb_dict()
                        )
                        
                        # Mark as processed
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
                        # Mark as error
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

            # Store metrics
            await self.store_metrics()
            
            if processed_count == 0:
                self.logger.warning(f"Batch complete but no posts were processed successfully")
            else:
                self.logger.info(f"Batch complete. Processed {processed_count} posts successfully")
            
            return processed_count

        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            return 0

    async def process_daemon(self):
        """Run the processing daemon"""
        self.logger.info(f"Starting processing daemon with version {self.version_id}")
        
        while True:
            try:
                processed = await self.process_batch()
                
                if processed == 0:
                    self.logger.info("No more posts to process, sleeping...")
                    await asyncio.sleep(30)
                else:
                    self.logger.info(f"Processed {processed} posts") 
                    await asyncio.sleep(10)
                    
            except Exception as e:
                self.logger.error(f"Daemon error: {str(e)}")
                await asyncio.sleep(60)

    def start(self):
        """Start the processing daemon"""
        asyncio.run(self.process_daemon())

if __name__ == '__main__':
    # Create the cleansing service
    service = EnhancedCleansingService(
        raw_table='Raw-Posts',
        cleansed_table='Cleansed-Posts',
        metrics_table='Cleansing-Metrics',
        batch_size=10
    )
    
    # Start processing
    service.start()
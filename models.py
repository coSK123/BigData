from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


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
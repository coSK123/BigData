from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum



class ProcessedFlag(Enum):
    ERROR = -1
    NOT_PROCESSED = 0
    PROCESSED = 1

class TwitterPost(BaseModel):
    id: str
    username: str
    content: str
    created_at: datetime
    retweet_count: int = Field(ge=0)
    like_count: int = Field(ge=0)
    reply_count: int = Field(ge=0)
    hashtags: Optional[List[str]]
    search_query: str
    keywords: Optional[List[str]]
    

    class Config:
        orm_mode = True
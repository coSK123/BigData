from dotenv import load_dotenv
import boto3
import os
import random
from datetime import datetime, timedelta
import uuid
from typing import List, Dict
import json

class MockDataGenerator:
    def __init__(self):
        load_dotenv()
        
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-north-1'
        )
        self.dynamodb = session.resource('dynamodb')
        self.cleansed_table = self.dynamodb.Table('Cleansed-Posts')
        
        # Mock data configurations
        self.candidates = ['trump', 'harris', 'biden']
        self.parties = ['republican', 'democrat']
        self.languages = ['en', 'es']
        self.devices = ['Twitter Web App', 'Twitter for iPhone', 'Twitter for Android', 'TweetDeck']
        self.states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        
        # Sample text templates
        self.text_templates = [
            "Just watched {candidate}'s speech. {sentiment} #Election2024",
            "The {party} party is {sentiment} for our country. {hashtag}",
            "{sentiment} about {candidate}'s policies on economy. {hashtag}",
            "Rally for {candidate} in {state} was {sentiment}! #Vote2024",
            "{candidate} speaks about important issues. {sentiment} #Politics"
        ]
        
        self.positive_words = ['great', 'amazing', 'excellent', 'impressive', 'strong']
        self.negative_words = ['concerning', 'disappointing', 'weak', 'problematic', 'wrong']
        self.hashtags = ['#Election2024', '#Vote2024', '#Politics', '#America', '#Democracy']

    def generate_mock_text(self) -> str:
        template = random.choice(self.text_templates)
        sentiment = random.choice(self.positive_words + self.negative_words)
        hashtag = random.choice(self.hashtags)
        candidate = random.choice(self.candidates)
        party = random.choice(self.parties)
        state = random.choice(self.states)
        
        return template.format(
            candidate=candidate,
            sentiment=sentiment,
            hashtag=hashtag,
            party=party,
            state=state
        )

    def generate_mock_post(self, date: datetime) -> Dict:
        # Randomly select candidate and matching party
        candidate = random.choice(self.candidates)
        party = 'republican' if candidate == 'trump' else 'democrat'
        
        return {
            'ID': str(uuid.uuid4()),
            'Date': date.isoformat(),
            'text': self.generate_mock_text(),
            'created_at': date.isoformat(),
            'processed_at': datetime.now().isoformat(),
            'candidates_mentioned': [candidate],
            'parties_mentioned': [party],
            'metrics': {
                'replies': random.randint(0, 1000),
                'retweets': random.randint(0, 5000),
                'favorites': random.randint(0, 10000),
                'views': random.randint(1000, 100000)
            },
            'author': {
                'id': str(random.randint(10000, 99999)),
                'verified': random.choice([True, False]),
                'follower_count': random.randint(100, 1000000)
            },
            'location': {
                'raw_location': f"{random.choice(self.states)}, USA",
                'state': random.choice(self.states)
            },
            'is_retweet': random.choice([True, False]),
            'language': random.choice(self.languages),
            'source_device': random.choice(self.devices)
        }

    def insert_mock_data(self, num_posts: int = 1000):
        # Generate posts over the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        posts = []
        for _ in range(num_posts):
            random_date = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            posts.append(self.generate_mock_post(random_date))
        
        # Insert in batches of 25 (DynamoDB limit)
        batch_size = 25
        for i in range(0, len(posts), batch_size):
            batch = posts[i:i + batch_size]
            with self.cleansed_table.batch_writer() as writer:
                for post in batch:
                    writer.put_item(Item=post)
            print(f"Inserted batch {i//batch_size + 1}/{len(posts)//batch_size + 1}")

def main():
    generator = MockDataGenerator()
    num_posts = 1000  # Adjust as needed
    
    print(f"Starting to generate {num_posts} mock posts...")
    generator.insert_mock_data(num_posts)
    print("Mock data generation complete!")

if __name__ == "__main__":
    main()
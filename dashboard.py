from .models import TwitterPost
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='eu-north-1'
)

dynamodb = session.resource('dynamodb')

# Replace 'your_table_name' with the name of your DynamoDB table
table = dynamodb.Table('Raw-Posts')




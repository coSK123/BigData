import boto3
import logging
from datetime import datetime
import asyncio
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostAwareProcessor:
    def __init__(self, table_name: str):
        self.cloudwatch = boto3.client('cloudwatch')
        self.table_name = table_name
        
    async def check_usage_within_limits(self) -> bool:
        """Check if current usage is within free tier limits"""
        try:
            metrics = self.cloudwatch.get_metric_data(
                MetricDataQueries=[
                    {
                        'Id': 'readCapacity',
                        'MetricStat': {
                            'Metric': {
                                'Namespace': 'AWS/DynamoDB',
                                'MetricName': 'ConsumedReadCapacityUnits',
                                'Dimensions': [
                                    {'Name': 'TableName', 'Value': self.table_name}
                                ]
                            },
                            'Period': 3600,
                            'Stat': 'Sum'
                        }
                    }
                ],
                StartTime=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
                EndTime=datetime.utcnow()
            )
            
            # Check if we're approaching free tier limits
            daily_usage = sum(metrics['MetricDataResults'][0]['Values'])
            return daily_usage < 20000  # 80% of free tier daily limit
            
        except Exception as e:
            logger.error(f"Error checking usage: {str(e)}")
            return False

    async def get_safe_batch_size(self) -> int:
        """Dynamically adjust batch size based on usage"""
        try:
            if await self.check_usage_within_limits():
                return 10  # Normal batch size when within limits
            else:
                return 5   # Reduced batch size when approaching limits
        except Exception as e:
            logger.error(f"Error getting safe batch size: {str(e)}")
            return 5  # Conservative batch size on error

# Usage example
async def main():
    processor = CostAwareProcessor('Raw-Posts')
    
    # Check if we're within free tier limits
    within_limits = await processor.check_usage_within_limits()
    if not within_limits:
        logger.warning("Approaching free tier limits!")
    
    # Get safe batch size
    batch_size = await processor.get_safe_batch_size()
    logger.info(f"Safe batch size: {batch_size}")

if __name__ == "__main__":
    asyncio.run(main())
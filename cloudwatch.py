import boto3
import json
from datetime import datetime, timedelta

class DynamoDBCostMonitor:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.dynamodb = boto3.client('dynamodb')
        
    def create_basic_alarms(self, table_name: str, max_daily_cost_usd: float = 1.0):
        """
        Create basic CloudWatch alarms for DynamoDB usage
        """
        # Create alarm for consumed read capacity
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{table_name}-ReadCapacityExceeded',
            AlarmDescription='Alert when read capacity approaches free tier limit',
            MetricName='ConsumedReadCapacityUnits',
            Namespace='AWS/DynamoDB',
            Dimensions=[
                {'Name': 'TableName', 'Value': table_name}
            ],
            Period=300,  # 5 minutes
            EvaluationPeriods=1,
            Threshold=20,  # Alert at 20 RCU (80% of free tier)
            ComparisonOperator='GreaterThanThreshold',
            Statistic='Sum',
            ActionsEnabled=True,
            # Add your SNS topic ARN here if you want notifications
            # AlarmActions=['arn:aws:sns:region:account-id:topic-name']
        )

        # Create alarm for consumed write capacity
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{table_name}-WriteCapacityExceeded',
            AlarmDescription='Alert when write capacity approaches free tier limit',
            MetricName='ConsumedWriteCapacityUnits',
            Namespace='AWS/DynamoDB',
            Dimensions=[
                {'Name': 'TableName', 'Value': table_name}
            ],
            Period=300,  # 5 minutes
            EvaluationPeriods=1,
            Threshold=20,  # Alert at 20 WCU (80% of free tier)
            ComparisonOperator='GreaterThanThreshold',
            Statistic='Sum',
            ActionsEnabled=True,
            # Add your SNS topic ARN here if you want notifications
            # AlarmActions=['arn:aws:sns:region:account-id:topic-name']
        )

    def get_current_usage(self, table_name: str):
        """
        Get current usage metrics for a table
        """
        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'readCapacity',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/DynamoDB',
                            'MetricName': 'ConsumedReadCapacityUnits',
                            'Dimensions': [
                                {'Name': 'TableName', 'Value': table_name}
                            ]
                        },
                        'Period': 3600,  # 1 hour
                        'Stat': 'Sum'
                    }
                },
                {
                    'Id': 'writeCapacity',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/DynamoDB',
                            'MetricName': 'ConsumedWriteCapacityUnits',
                            'Dimensions': [
                                {'Name': 'TableName', 'Value': table_name}
                            ]
                        },
                        'Period': 3600,  # 1 hour
                        'Stat': 'Sum'
                    }
                }
            ],
            StartTime=datetime.utcnow() - timedelta(hours=24),
            EndTime=datetime.utcnow()
        )
        
        return response

def setup_cost_monitoring():
    """Set up cost monitoring for DynamoDB tables"""
    monitor = DynamoDBCostMonitor()
    
    # Create alarms for Raw-Posts table
    monitor.create_basic_alarms('Raw-Posts')
    
    # Create alarms for Cleansed-Posts table
    monitor.create_basic_alarms('Cleansed-Posts')
    
    print("Cost monitoring has been set up successfully!")
    print("CloudWatch alarms will trigger if usage approaches free tier limits")

if __name__ == '__main__':
    # setup_cost_monitoring() # done
    pass
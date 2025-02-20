import boto3
import json

import os

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

if not AWS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables")

bedrock_client = boto3.client(service_name='bedrock',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
        )

response = bedrock_client.list_foundation_models(byProvider="meta")

for summary in response["modelSummaries"]:
        print(summary["modelId"])
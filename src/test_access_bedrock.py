import boto3
import json



bedrock_client = boto3.client(service_name='bedrock',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
        )

response = bedrock_client.list_foundation_models(byProvider="meta")

for summary in response["modelSummaries"]:
        print(summary["modelId"])
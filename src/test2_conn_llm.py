import boto3
import botocore.config
import json
import os

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

if not AWS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables")

def blog_generate_using_bedrock(blogtopic:str)-> str:
    prompt=f"""<s>[INST]Human: Write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]
    """

    body={
        "prompt":prompt,
        "max_gen_len":512,
        "temperature":0.5,
        "top_p":0.9
    }

    try:
        #model_id = "meta.llama3-3-70b-instruct-v1:0"
        model_id = "us.meta.llama3-1-70b-instruct-v1:0"
        model_id = "arn:aws:bedrock:us-east-1:816795924256:inference-profile/us.meta.llama3-1-70b-instruct-v1:0"
        bedrock=boto3.client("bedrock-runtime",
                             region_name="us-east-1",
                             config=botocore.config.Config(read_timeout=300,retries={'max_attempts':3}), 
                             aws_access_key_id=AWS_KEY, 
                             aws_secret_access_key=AWS_SECRET_KEY)
        response=bedrock.invoke_model(body=json.dumps(body),modelId=model_id)

        response_content=response.get('body').read()
        response_data=json.loads(response_content)
        print(response_data)
        blog_details=response_data['generation']
        return blog_details
    except Exception as e:
        print(f"Error generating the blog:{e}")
        return "ERROR"

blog_generate_using_bedrock("Machine Learning")
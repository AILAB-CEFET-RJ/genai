import boto3
import json
import os

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

if not AWS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables")

bedrock = boto3.client(service_name='bedrock', 
                       region_name='us-east-1', 
                       aws_access_key_id=AWS_KEY, 
                       aws_secret_access_key=AWS_SECRET_KEY)

response = bedrock.list_foundation_models(byProvider="meta")

for summary in response["modelSummaries"]:
    print(summary["modelId"])

# modelId obtained from printing out modelIds in previous step
# modelId = 'anthropic.claude-v2:1'
modelId = 'meta.llama3-3-70b-instruct-v1:0'

### parameters for the LLM to control text-generation
# temperature increases randomness as it increases
temperature = 0.5
# top_p increases more word choice as it increases
top_p = 1
# maximum number of tokens togenerate in the output
max_tokens_to_generate = 250

system_prompt = "All your responses should be in Haiku form"

messages = [{"role": "user", "content": "Hello, world, tell me a poem"}, 
            {"role": "assistant", "content": "Here is a haiku poem for you:\n\nLaughing out loud\nAt silly jokes and stories\nBringing joyful smiles"},
            {"role": "user", "content": "I don't want to smile"}]

bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
region_name='us-east-1', 
aws_access_key_id=AWS_KEY, 
aws_secret_access_key=AWS_SECRET_KEY)

body = json.dumps({
            "messages": messages,
            "system": system_prompt,
            "max_tokens": max_tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p#,
            # "anthropic_version": "bedrock-2023-05-31"
})

response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept="application/json", contentType="application/json")

response_body = json.loads(response.get('body').read())
result = response_body.get('content', '')
print(result)
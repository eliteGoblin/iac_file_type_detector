from openai import OpenAI
import os
from icecream import ic
import re
import shutil
from datetime import datetime


client = OpenAI(api_key=os.getenv("GPT_KEY"))

# Define available services for CloudFormation and Terraform
aws_services_with_cloudformation_support = [
    "ec2", "s3", "vpc", "lambda", "dynamodb", "rds", "cloudfront",
    "sns", "sqs", "iam", "elasticache", "kinesis", "eks", "ecs",
    "elb", "alb", "route53", "cloudwatch", "athena", "glue",
    "apigateway", "batch", "cloudformation", "codebuild", "codedeploy",
    "codepipeline", "config", "dms", "emr", "elasticbeanstalk",
    "fargate", "firehose", "kms", "logs", "organizations",
    "sagemaker", "secretsmanager", "ses", "shield", "waf"
]

# Define available services for Azure ARM
azure_services_with_arm_support = [
    "virtualMachines", "storageAccounts", "virtualNetworks", "functions", "cosmosDB",
    "sqlDatabases", "appServices", "keyVaults", "loadBalancers", "applicationGateways",
    "containerInstances", "kubernetesServices", "appServicePlans", "redisCaches",
    "networkSecurityGroups", "trafficManagerProfiles", "vpnGateways", "dnsZones",
    "eventHubs", "serviceBusNamespaces"
]


def find_first_char_after_pattern(text):
    # Compile a regular expression to match the pattern like ```json\n, ```yml\n, etc.
    regex = re.compile(r'```.*?\n')
    
    # Search for the pattern in the text
    match = regex.search(text)
    
    if match:
        # The position of the first character after the matched pattern
        return match.end()
    else:
        # If the pattern is not found, return -1 or any other indication of failure
        return -1

def generate_one_iac_file(service, iac_type, model:str, format):
    prompt = f"""
      Generate a {iac_type} configuration for {service} in {format} format. Note must in this {format} format!!!
      with real values. Code should be valid and complete(i.e not contain place holder value), minimum 500 words
      contain related resources close to production-ready, e.g when create EC2, contain VPC, subnet, security group, etc.
      do NOT contain any comments, just real code
      """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    code = response.choices[0].message.content.strip()
    code_start = find_first_char_after_pattern(code)
    code_end = code.find('```', code_start)

    if code_start != -1 and code_end != -1:
        cleaned_code = code[code_start:code_end].strip()
    else:
        cleaned_code = code  # Fallback if no code block markers are found

    return cleaned_code


def verify_iac_input(iac_type, format):
    valid_formats = {
        "cloudformation": ["json", "yml"],
        "terraform": ["json", "tf"],
        "ansible": ["yml"],
        "azure_arm": ["json", "yml"]
    }

    if iac_type not in valid_formats.keys():
        raise ValueError("Invalid IAC type specified.")
    if format not in valid_formats[iac_type]:
        raise ValueError("Invalid format specified.")


def mkdir_target_dir(output_base: str, iac_type: str, model: str, format: str) -> str:
    """
    structure is {output_dir}/{iac_type}/{format}/{GPT Model}/{Timestamp to minute}/{type_specific_prefix}_{i}.yml
    iac_type is one of cloudformation, terraform, ansible, arm
    """
    target_dir = os.path.join(output_base, iac_type, format, model, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir
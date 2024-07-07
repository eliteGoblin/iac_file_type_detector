import click
import os
from ansible import generate_ansible_code_concurrent
from cloudformation import generate_cloudformation_code_concurrent
from azure_arm import generate_azure_arm_code_concurrent
from noniac import generate_noniac_code_concurrent
from generate_iac import aws_services_with_cloudformation_support, azure_services_with_arm_support, verify_iac_input
from icecream import ic

output_root = "./gitignore/iac_files"

@click.group()
def cli():
    """Command Line Interface for various IAC tools and Non-IAC options"""
    pass

@cli.command()
@click.option('--file-format', type=click.Choice(['yml', 'json']), required=True, help='Specify the file format for CloudFormation')
@click.option('--count-per-service', type=int, default=1, required=True, help='Specify the number of files to generate per service')
def cloudformation(file_format, count_per_service):
    """CloudFormation specific commands"""
    verify_iac_input("cloudformation", file_format)
    target_dir = generate_cloudformation_code_concurrent(
        output_base=output_root,
        model="gpt-3.5-turbo",
        aws_service=aws_services_with_cloudformation_support,
        format=file_format,
        count_per_service=count_per_service,
    )
    print(f"CloudFormation selected. Number of files to generate per service: {count_per_service}, target directory: {target_dir}")

@cli.command()
@click.option('--num-files', type=int, default=1, help='Number of Ansible files to generate')
def ansible(num_files):
    """Ansible specific commands"""
    ic("here")
    click.echo(f"Ansible selected. Number of files to generate: {num_files}")
    generate_ansible_code_concurrent(
        output_base=output_root,
        model="gpt-3.5-turbo",
        num_files=num_files, 
        concurrency=10)

@cli.command()
@click.option('--provider', type=str, default='aws', help='terraform provider, e.g aws, azure, gcp, etc.')
@click.option('--num-files-per-service', type=int, default=1, help='Number of Terraform files to generate per service, e.g AWS s3')
def terraform():
    """Terraform specific commands"""
    # skip for now, since terraform using .tf to recognize
    click.echo("Terraform selected")

@cli.command()
@click.option('--file-format', type=click.Choice(['yml', 'json']), required=True, help='Specify the file format for Azure ARM')
@click.option('--count-per-service', type=int, default=1, required=True, help='Specify the number of files to generate per service')
def azure_arm(file_format, count_per_service):
    """Azure ARM specific commands"""
    verify_iac_input("azure_arm", file_format)
    target_dir = generate_azure_arm_code_concurrent(
        output_base=output_root,
        model="gpt-3.5-turbo",
        azure_service=azure_services_with_arm_support,
        format=file_format,
        count_per_service=count_per_service,
    )
    print(f"CloudFormation selected. Number of files to generate per service: {count_per_service}, target directory: {target_dir}")

@cli.command()
@click.option('--num-files', type=int, default=1, help='Number of Ansible files to generate')
@click.option('--file-format', type=click.Choice(['json', 'yml']), help='Specify the file format for Non-IAC')
def noniac(file_format, num_files):
    """Non-IAC specific commands"""
    click.echo(f"Non-IAC selected with file format: {file_format}")
    generate_noniac_code_concurrent(
        output_base=output_root,
        model="gpt-3.5-turbo",
        num_files=num_files,
        concurrency=10
    )

if __name__ == '__main__':
    cli()

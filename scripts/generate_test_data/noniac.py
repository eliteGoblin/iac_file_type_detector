
from generate_iac import client, find_first_char_after_pattern, mkdir_target_dir
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

common_noniac_yaml_types = [
    "Docker Compose",
    "Kubernetes Configuration",
    "GitHub Actions Workflow",
    "CircleCI Configuration",
    "Travis CI Configuration",
    "Azure DevOps Pipeline",
    "GitLab CI Pipeline",
    "Drone CI Configuration",
    "Ansible Playbook",
    "Heroku App Manifest"
]


def generate_noniac_code_concurrent(output_base: str, model: str, num_files: int, concurrency: int = 10) -> str:
    target_dir = mkdir_target_dir(
        output_base=output_base, 
        iac_type="noniac", 
        model=model, 
        format="yml")
    
    prompt = """
      Generate NonIAC files(infra as code), with format {format}, file with real values.
      NonIAC means files with valid {format} content, but not in IAC format, e.g NOT in CloudFormation, Terraform, Ansible, Azure ARM etc.
      Remember, NOT in IAC content, while in {format} format, 
      make it common software engineer/devops files in {format}, but NOT to create infra.
      example: docker-compose, k8s, helm, GHA, CI, Azure devops, Jenkins, {common_noniac_yaml_types}, custom app config yml(deep nested)etc. 
      Make it random each time you provide me answer.
      At least 150 lines of code, close to production code, do NOT contain any comments, just real code
    """

    prompt = """
      make it common software engineer/devops files in {format}, but NOT to create infra, i.e not in terraform, cloudformation, ansible, azure arm etc.
      example: custom app config yml(deep nested)etc, for own app config, DB config, system config, like vault etc. 
      Make it random each time you provide me answer.
      At least 150 lines of code, close to production code, do NOT contain any comments, just real code
    """

    def process_task(i):
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
        code_end = code.rfind('```') if '```' in code else len(code)
        if code_start != -1 and code_end != -1:
            cleaned_code = code[code_start:code_end].strip()
        else:
            cleaned_code = code
        file_path = os.path.join(target_dir, f"noniac_{i}.yaml")
        with open(file_path, "w") as file:
            file.write(cleaned_code)
        return file_path

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_task, i): i for i in range(num_files)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            i = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"File {i} generated an exception: {exc}")
            else:
                print(f"File {i} processing complete")
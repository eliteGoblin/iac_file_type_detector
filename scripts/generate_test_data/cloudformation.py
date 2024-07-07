from generate_iac import generate_one_iac_file, mkdir_target_dir
from tqdm import tqdm
from icecream import ic
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_cloudformation_code(
        output_base, model: str, aws_service: list[str], format: str, count_per_service: int = 1) -> str:

    target_dir = mkdir_target_dir(output_base, "cloudformation", model, format)
    for service in tqdm(aws_service, desc=f"Generating cloudformation files"):
        for i in range(count_per_service):
            code = generate_one_iac_file(
                service=service,
                iac_type="cloudformation",
                model=model,
                format=format
            )
            # ic(code)
            with open(os.path.join(
                target_dir,
                f"cfn_{service}_{i}.{format}"), "w") as file:
                file.write(code)

    return target_dir

def generate_cloudformation_code_concurrent(
        output_base, model: str, aws_service: list[str], format: str, 
        count_per_service: int = 1, 
        concurrency: int = 10) -> str:

    target_dir = mkdir_target_dir(output_base, "cloudformation", model, format)

    def process_task(service, i):
        code = generate_one_iac_file(
            service=service,
            iac_type="cloudformation",
            model=model,
            format=format
        )
        file_path = os.path.join(target_dir, f"cfn_{service}_{i}.{format}")
        with open(file_path, "w") as file:
            file.write(code)
        return file_path

    tasks = [(service, i) for service in aws_service for i in range(count_per_service)]
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_task, service, i): (service, i) for service, i in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating CloudFormation files"):
            service, i = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Service {service}, file {i} generated an exception: {exc}")
            else:
                print(f"Service {service}, file {i} processing complete")

    return target_dir
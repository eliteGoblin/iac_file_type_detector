from generate_iac import generate_one_iac_file, mkdir_target_dir
from tqdm import tqdm
from icecream import ic
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_azure_arm_code_concurrent(
        output_base, model: str, azure_service: list[str], format: str, 
        count_per_service: int = 1, 
        concurrency: int = 10) -> str:

    target_dir = mkdir_target_dir(output_base, "azure_arm", model, format)

    def process_task(service, i):
        code = generate_one_iac_file(
            service=service,
            iac_type="azure arm",
            model=model,
            format=format
        )
        file_path = os.path.join(target_dir, f"azure_arm_{service}_{i}.{format}")
        with open(file_path, "w") as file:
            file.write(code)
        return file_path

    tasks = [(service, i) for service in azure_service for i in range(count_per_service)]
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_task, service, i): (service, i) for service, i in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Azure ARM files"):
            service, i = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Service {service}, file {i} generated an exception: {exc}")
            else:
                print(f"Service {service}, file {i} processing complete")

    return target_dir
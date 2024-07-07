
from generate_iac import client, find_first_char_after_pattern, mkdir_target_dir
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_ansible_code(output_base: str, model: str, num_files: int):

    target_dir = mkdir_target_dir(output_base, "ansible", model, "yml")
    
    prompt = """
      Generate a common Ansible playbook with yml format, with real values.
      Ansible playbook ONLY include VM that agasint CIS security best practice.
      e.g contain Ansible code that has certain securtiy vulneratbilities.
      Make it random each time you provide me answer. 
      Close to production code, do NOT contain any comments, just real code
    """

    for i in tqdm(range(num_files), desc="Processing files"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        code = response.choices[0].message.content.strip()
        # Find the start of the code block and strip the content before and after it
        code_start = find_first_char_after_pattern(code)
        code_end = code.rfind('```') if '```' in code else len(code)
        if code_start != -1 and code_end != -1:
            cleaned_code = code[code_start:code_end].strip()
        else:
            cleaned_code = code
        file_path = os.path.join(target_dir, f"ansible_playbook_{i}.yaml")
        with open(file_path, "w") as file:
            file.write(cleaned_code)

def generate_ansible_code_concurrent(output_base: str, model: str, num_files: int, concurrency: int = 10) -> str:
    target_dir = mkdir_target_dir(output_base, "ansible", model, "yml")
    
    prompt = """
      Generate a common Ansible playbook with yml format, with real values.
      Ansible playbook ONLY include VM that agasint CIS security best practice.
      e.g contain Ansible code that has certain securtiy vulneratbilities.
      Make it random each time you provide me answer. 
      Close to production code, do NOT contain any comments, just real code
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
        file_path = os.path.join(target_dir, f"ansible_playbook_{i}.yaml")
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
import os
import json
import logging
import sys
from contextlib import contextmanager

def get_case_path(output_root, pde_type, project_name):
    """
    Generates and returns the next case folder path:
    ./outputs/PDE_<type>/<project>/case_NNN/
    """
    pde_dir = os.path.join(output_root, f"PDE_{pde_type}")
    project_dir = os.path.join(pde_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Find next case number
    existing_cases = [d for d in os.listdir(project_dir) if d.startswith("case_") and os.path.isdir(os.path.join(project_dir, d))]
    
    if not existing_cases:
        next_num = 1
    else:
        # Extract numbers, find max, and add 1
        nums = []
        for d in existing_cases:
            try:
                nums.append(int(d.split("_")[1]))
            except:
                pass
        next_num = max(nums) + 1 if nums else 1
        
    case_folder = f"case_{next_num:03d}"
    case_path = os.path.join(project_dir, case_folder)
    os.makedirs(case_path, exist_ok=True)
    return case_path

def save_config_json(config, folder, filename):
    """
    Saves a configuration dictionary to a JSON file.
    """
    # Filter out non-serializable objects just in case
    serializable_config = {}
    for k, v in config.items():
        if isinstance(v, (dict, list, str, int, float, bool, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)
            
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(serializable_config, f, indent=4)

@contextmanager
def redirect_stdout_to_file(file_path):
    """
    Context manager to redirect all stdout and stderr to a file.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(file_path, 'a', encoding='utf-8') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

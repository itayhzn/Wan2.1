from datetime import datetime
import sys
import torch
import os

def encode_params(prompt, task, size, ulysses_size, ring_size, addit_prompt=None, experiment_name=None):
    def escape(s):
        return s.replace(" ", "_").replace("/", "_").replace(",", "_") \
                 .replace("'", "_").replace('"', "_")[:30]
    
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_file = ""

    if experiment_name:
        save_file += f"{experiment_name}_"

    save_file += f"{task}_{size.replace('*','x') if sys.platform=='win32' else size}_{ulysses_size}_{ring_size}_{escape(prompt)}_{formatted_time}"
    
    if addit_prompt:
        save_file += "_ADDIT_" + escape(addit_prompt)

    return save_file

def save_tensors(self, save_tensors_dir, tensors_dict):
    r"""
    Save tensors to disk for debugging purposes.
    """
    # mkdir if not exists
    if not os.path.exists(save_tensors_dir):
        os.makedirs(save_tensors_dir)
        
    print(f'======= Saving tensors to {save_tensors_dir}')
    for name, tensor in tensors_dict.items():
        print(f'\tSaving tensor {name} to {os.path.join(save_tensors_dir, name)}')
        torch.save(tensor.clone(), os.path.join(save_tensors_dir, f'{name}.pt'))
    print(f'======= Saved tensors to {save_tensors_dir}')
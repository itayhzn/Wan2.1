from datetime import datetime
import sys
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import uuid

def encode_params(prompt, seed, task=None, size=None, ulysses_size=None, ring_size=None, experiment_name=None):
    def escape(s):
        return s.replace(" ", "_").replace("/", "_").replace(",", "_") \
                 .replace("'", "_").replace('"', "_")[:60]
    
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_file = f"{formatted_time}_"

    if experiment_name:
        save_file += f"{experiment_name}_"
    if task:
        save_file += f"{task}_"
    if size:
        save_file += f"{size.replace('*', 'x')}_" if sys.platform == 'win32' else f"{size}_"
    if ulysses_size:
        save_file += f"{ulysses_size}_"
    if ring_size:
        save_file += f"{ring_size}_"
    if prompt:
        save_file += f"{escape(prompt)}_"
    if seed:
        save_file += f"{seed}_"
    
    return save_file

def save_tensors(save_tensors_dir, tensors_dict):
    r"""
    Save tensors to disk for debugging purposes.
    """
    # mkdir if not exists
    if not os.path.exists(save_tensors_dir):
        os.makedirs(save_tensors_dir)

    print(f'======= Saving tensors to {save_tensors_dir}')
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f'\tSaving tensor {name} to {os.path.join(save_tensors_dir, name)}')
            torch.save(tensor, os.path.join(save_tensors_dir, f'{name}.pt'))
        else:
            print(f'{name} is not a tensor, skipping save. Type: {type(tensor)}')
    print(f'======= Saved tensors to {save_tensors_dir}')

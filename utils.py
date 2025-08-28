from datetime import datetime
import sys
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import uuid

def encode_params(args):
    def escape(s):
        return s.replace(" ", "_").replace("/", "_").replace(",", "_") \
                 .replace("'", "_").replace('"', "_")[:60]
    
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    s = f"{formatted_time}_"

    if args.experiment_name:
        s += f"{args.experiment_name}_"
    if args.prompt:
        s += f"{escape(args.prompt)}_"
    if args.base_seed:
        s += f"{args.base_seed}_"
    if args.optimization_iterations:
        s += f"it={args.optimization_iterations}_"
    if args.optimization_lr:
        s += f"lr={args.optimization_lr}_"
    if args.optimization_start_step:
        s += f"start={args.optimization_start_step}_"
    if args.optimization_end_step:
        s += f"end={args.optimization_end_step}_"
    if args.loss_name:
        s += f"loss={args.loss_name}_"

    if s.endswith("_"):
        s = s[:-1]

    return s

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

def log_losses(filename, losses, step, dirname='logs'):
    """
    Log losses to console and file.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    log_file = os.path.join(dirname, filename)
    with open(log_file, 'a') as f:
        f.write(f"Step {step}:\n")
        for name, value in losses.items():
            f.write(f" {name}: {value.item()}\n")
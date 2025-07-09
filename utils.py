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

def save_tensors(save_tensors_dir, tensors_dict):
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


def compute_subject_mask(q, k_context, subject_token_index):
    """
    Compute a mask for the subject token in the attention map.
    q: Tensor of shape [B, L1, num_heads, d]
    k_context: Tensor of shape [B, L2, num_heads, d]
    """
    # Create a mask where the subject token is 1 and all others are 0
    # take only head 3 
    q = q.clone()[0, :, 3, :]  # [L1, d]
    k_context = k_context.clone()[0, subject_token_index, 3, :].unsqueeze(0)  # [1, d]
    # compute attention map for the subject token
    attention_map = q @ k_context.transpose(-2, -1) # [L1, 1]
    # get mask with softmax
    subject_mask = attention_map.softmax(dim=-1).unsqueeze(-1).unsqueeze(0) # [1, L1, 1, 1]
    return subject_mask
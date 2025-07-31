from datetime import datetime
import sys
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import uuid

def encode_params(prompt, task, size, ulysses_size, ring_size, edit_prompt=None, subject_prompt=None, experiment_name=None):
    def escape(s):
        return s.replace(" ", "_").replace("/", "_").replace(",", "_") \
                 .replace("'", "_").replace('"', "_")[:30]
    
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_file = f"{formatted_time}_"

    if experiment_name:
        save_file += f"{experiment_name}_"

    save_file += f"{task}_{size.replace('*','x') if sys.platform=='win32' else size}_{ulysses_size}_{ring_size}_{escape(prompt)}"
    
    if edit_prompt:
        save_file += "_SUBJECT_" + escape(subject_prompt) + "_EDIT_" + escape(edit_prompt)

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
            t = tensor.cpu().clone()
        else:
            t = torch.Tensor(tensor, device='cpu').clone()
        print(f'\tSaving tensor {name} to {os.path.join(save_tensors_dir, name)}')
        torch.save(t, os.path.join(save_tensors_dir, f'{name}.pt'))
    print(f'======= Saved tensors to {save_tensors_dir}')

def read_video(video_path):
    """
    Read mp4 video and return a list of frames as numpy arrays.
    """
    video = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        # frame dim [H, W, C]
        video.append(frame)
    cap.release()
    return video

def save_video_tensor_in_dir(video, output_dir):
    """
        video: [F, H, W, C]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    # normalize video to 0-255
    video = (video - video.min()) / (video.max() - video.min()) * 255.0
    video = video.astype(np.uint8)  # Convert to uint8 for saving as images

    for i, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(output_dir, f"{i:04d}.jpg"))

def delete_video_dir(video_dir):
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

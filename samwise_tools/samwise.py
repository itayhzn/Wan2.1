'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from util import misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
import sys
from pycocotools import mask as cocomask
from tools.colormap import colormap
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from tools.metrics import db_eval_boundary, db_eval_iou
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
from datasets.transform_utils import vis_add_mask
from types import SimpleNamespace

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def build_samwise_model(args=None):
    
    if not args:
        args = get_samwise_args()

    pretrained_model_link = 'https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing'
    assert os.path.isfile(args.resume), f"You should download the model checkpoint first. Run 'cd pretrain &&  gdown --fuzzy {pretrained_model_link}"

    model = build_samwise(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    if list(checkpoint['model'].keys())[0].startswith('module'):
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
    checkpoint = on_load_checkpoint(model, checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


def extract_frames_from_mp4(video_path):
    extract_folder = 'frames_' + os.path.basename(video_path).split('.')[0]
    print(f'Extracting frames from .mp4 in {extract_folder} with ffmpeg...')
    if os.path.isdir(extract_folder):
        print(f'{extract_folder} already exists, using frames in that folder')
    else:
        os.makedirs(extract_folder)
        extract_cmd = "ffmpeg -i {in_path} -loglevel error -vf fps=10 {folder}/frame_%05d.png"
        ret = os.system(extract_cmd.format(in_path=video_path, folder=extract_folder))
        if ret != 0:
            print('Something went wrong extracting frames with ffmpeg')
            sys.exit(ret)
    frames_list=os.listdir(extract_folder)
    frames_list = sorted([os.path.splitext(frame)[0] for frame in frames_list])

    return extract_folder, frames_list, '.png'


def compute_masks(model, text_prompt, frames_folder, frames_list, ext, args):
    all_pred_masks = []
    vd = VideoEvalDataset(frames_folder, frames_list, ext=ext)
    dl = DataLoader(vd, batch_size=args.eval_clip_window, num_workers=args.num_workers, shuffle=False)
    origin_w, origin_h = vd.origin_w, vd.origin_h
    # 3. for each clip
    for imgs, clip_frames_ids in tqdm(dl):
        clip_frames_ids = clip_frames_ids.tolist()
        imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size, 'frame_ids': clip_frames_ids}

        with torch.no_grad():
            outputs = model([imgs], [text_prompt], [target])

        pred_masks = outputs["pred_masks"]  # [t, q, h, w]
        pred_masks = pred_masks.unsqueeze(0)
        pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()
        all_pred_masks.append(pred_masks)

    # store the video results
    all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()  # (video_len, h, w)

    return all_pred_masks


def inference(args, model, save_path_prefix, in_path, text_prompts):
    # load data
    if os.path.isfile(in_path) and not args.image_level:
        frames_folder, frames_list, ext = extract_frames_from_mp4(in_path)
    elif os.path.isfile(in_path) and args.image_level:
        fname, ext = os.path.splitext(in_path)
        frames_list = [os.path.basename(fname)]
        frames_folder = os.path.dirname(in_path)
    else:
        frames_folder = in_path
        frames_list = sorted(os.listdir(frames_folder))
        ext = os.path.splitext(frames_list[0])[1]
        frames_list = [os.path.splitext(frame)[0] for frame in frames_list if os.path.splitext(frame)[1] == ext]
        
    model.eval()
    print(f'Begin inference on {len(frames_list)} frames')
    # For each expression
    for i in range(len(text_prompts)):
        text_prompt = text_prompts[i]

        all_pred_masks = compute_masks(model, text_prompt, frames_folder, frames_list, ext, args)
            
        save_visualize_path_dir = join(save_path_prefix, text_prompt.replace(' ', '_'))
        os.makedirs(save_visualize_path_dir, exist_ok=True)
        print(f'Saving output to disk in {save_visualize_path_dir}')
        out_files_w_mask = []
        for t, frame in enumerate(frames_list):
            # original
            img_path = join(frames_folder, frame + ext)
            source_img = Image.open(img_path).convert('RGBA') # PIL image

            # draw mask
            source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i%len(color_list)])
            # save
            save_visualize_path = join(save_visualize_path_dir, frame + '.png')
            source_img.save(save_visualize_path)
            out_files_w_mask.append(save_visualize_path)

        if not args.image_level:
            # Create the video clip from images
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip(out_files_w_mask, fps=10)
            # Write the video file
            clip.write_videofile(join(save_path_prefix, text_prompt.replace(' ', '_')+'.mp4'), codec='libx264')

    print(f'Output masks and videos can be found in {save_path_prefix}')
    return 

def get_samwise_args(args=None):
    defaults = {
        # Training hyperparameters
        "lr": 1e-5,
        "batch_size": 2,
        "batch_size_val": 1,
        "num_frames": 8,
        "weight_decay": 0,
        "epochs": 6,
        "lr_drop": [60000],
        "clip_max_norm": 1,

        # Image Encoder: SAM2
        "sam2_version": "base",
        "disable_pred_obj_score": False,
        "motion_prompt": False,
        "image_level": False,  # If True, the model will run on a single image instead of a video clip

        # Cross Modal Temporal Adapter settings
        "HSA": True,
        "HSA_patch_size": [8, 4, 2],
        "adapter_dim": 256,
        "fusion_stages_txt": [4, 8, 12],
        "fusion_stages": [1, 2, 3],

        # Conditional Memory Encoder (CME) settings
        "use_cme_head": False,
        "switch_mem": "reweight",
        "cme_decision_window": 4,

        # Dataset settings
        "dataset_file": "ytvos",
        "coco_path": "data/coco",
        "ytvos_path": "data/ref-youtube-vos",
        "davis_path": "data/ref-davis",
        "mevis_path": "data/MeViS_release",
        "max_size": 1024,
        "augm_resize": False,

        # General settings
        "output_dir": "samwise_output",
        "name_exp": "default",
        "device": "cuda",
        "seed": 0,
        "resume": 'pretrain/final_model_mevis.pth',
        "resume_optimizer": False,
        "start_epoch": 0,
        "eval": False,
        "num_workers": 0,
        "no_distributed": False,

        # Testing and evaluation settings
        "threshold": 0.5,
        "split": "valid",
        "visualize": False,
        "eval_clip_window": 8,
        "set": "val",
        "task": "unsupervised",
        "results_path": None
    }

    # combine defaults with provided args
    if args is not None:
        for key, value in args.items():
            defaults[key] = value

    defaults['output_dir'] = os.path.join(defaults['output_dir'], defaults['name_exp'])

    return SimpleNamespace(**defaults)

def read_video(vid_folder, frames_list, ext='.png'):
    """
    Read video frames from a folder and return a list of frames as numpy arrays.
    """
    vd = VideoEvalDataset(vid_folder, frames_list, ext=ext)
    return [frame for frame in vd]

__all__ = [
    'get_samwise_args',
    'build_samwise_model',
    'extract_frames_from_mp4',
    'compute_masks',
    'inference',
]
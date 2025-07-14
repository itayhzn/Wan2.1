import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from sam2.sam2_video_predictor import SAM2VideoPredictor
from wan.modules.vae import WanVAE
import uuid

from utils import save_video_tensor_in_dir, delete_video_dir

# create a class to handle the segmentor logic
class LatentSegmentor:
    def __init__(self, vae=None, sam2=None, device='cuda'):
        self.sam2 = sam2 if sam2 is not None else SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
        self.vae = vae if vae is not None else WanVAE(
            vae_pth=os.path.join('Wan2.1-T2V-1.3B', 'Wan2.1_VAE.pth'),
            device=torch.device(device)
            )
        self.state_initialized = False

    def compute_subject_mask(self, latents, points, labels):
        """
        Run latent through VAE to get video frames.
        Then, use compute_subject_mask to get the subject masks.
        Finally, downsample the masks to match the latent resolution.
        Args:
            latents: [1, c, f, h, w]
            points: np.array of shape [N, 2] where N is the number of points
                    each point is in the format [x, y]
            labels: np.array of shape [N] where N is the number of points
                    each label is 1 for positive point, 0 for negative point
        Returns:
            masks:  dict of shape {frame_idx: mask} where mask is a numpy array of shape [h, w]
                    1 for subject, 0 for background
        """
        # decode the latent to get video frames and save them in dir
        videos = self.vae.decode(latents)  # [1, c, f, h, w]
        video = videos[0]  # [c, f, h, w]
        video_dir = str(uuid.uuid4().hex)  # create a unique directory name
        
        # translate points from latent space to video space
        latent_height, latent_width, latent_frame_cnt = latents[0].shape[-2], latents[0].shape[-1], latents[0].shape[-3]
        video_height, video_width, video_frame_cnt = video.shape[-2], video.shape[-1], video.shape[-3]
        points = (points * np.array([1.0 * video_width / latent_width, 1.0 * video_height / latent_height])).astype(int)    
        
        save_video_tensor_in_dir(video.permute(1,2,3,0), video_dir)
        del videos, video  # free memory
        subject_masks = self._compute_subject_mask(video_dir, points, labels)
        delete_video_dir(video_dir)

        # downsample the masks to match latent resolution
        downsampled_masks = {}
        idx = 0
        for frame_idx, mask in subject_masks.items():
            if frame_idx % (video_frame_cnt // latent_frame_cnt) == 0: # downsampling F -> f
                continue
            # downsampling H,W -> h,w
            downsampled_mask = cv2.resize(mask.astype(np.float32), (latents[0].shape[-1], latents[0].shape[-2]), interpolation=cv2.INTER_LINEAR)
            downsampled_masks[idx] = downsampled_mask
            idx += 1

        return downsampled_masks
    
    def _compute_subject_mask(self, video_dir, points, labels):
        if self.state_initialized:
            self.sam2.reset_state(video_dir)
        else:
            inference_state = self.sam2.init_state(video_dir)
            self.state_initialized = True
        
        frame_idx = 0
        obj_id = 1

        _,  out_obj_ids, out_mask_logits = self.sam2.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        subject_masks = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2.propagate_in_video(inference_state):
            # Find the mask for our specific object ID
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id == obj_id:
                    subject_masks[out_frame_idx] = np.squeeze((out_mask_logits[i] > 0.0).cpu().numpy())
                    break

        return subject_masks
    
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
        self.vae = vae
        self.inference_state = None
        self.masks = None

    def reset_inference_state(self):
        self.inference_state = None
        self.masks = None

    def get_precomputed_masks(self):
        """
        Returns the precomputed masks if they exist, otherwise returns None.
        """
        return self.masks

    def sample_points(self, first_frame_map, w, h):
        first_frame_map = first_frame_map.clone()
        max_value = first_frame_map.max()
        first_frame_map = first_frame_map[first_frame_map > 0.35 * max_value]  # filter out low attention points
        points = []
        while len(points) < 4 and first_frame_map.numel() > 0:
            positive_point_index = torch.argmax(first_frame_map).item()
            pos_i, pos_j = divmod(positive_point_index, w.item())  # [h, w]
            points.append([pos_j, pos_i])
            
            # remove the point and its neighborhood from the map
            for i in range(max(0, pos_i - 1), min(h.item(), pos_i + 2)):
                for j in range(max(0, pos_j - 1), min(w.item(), pos_j + 2)):
                    first_frame_map[i, j] = 0

        return torch.Tensor(points, dtype=torch.int16)  # [N, 2] where N is the number of points sampled

    def compute_subject_mask(self, x, q, k, grid_sizes):
            if self.masks is not None:
                return self.masks
            
            _, F, H, W = x.shape
            f, h, w = grid_sizes[0]

            stride = torch.tensor([F//f, H // h, W // w], dtype=torch.int64)

            # compute subject mask
            q_1_3 = q[0, :, 3, :] # [L1, d]
            k_subject_3 = k[0, :, 3, :] # [L2, d]
            attention_map = q_1_3 @ k_subject_3.transpose(-2, -1)  # [L1, L2]
            attention_map = attention_map[:, 0] # [L1] 
            
            # reshape to [f, h, w]
            attention_map = attention_map.view(grid_sizes[0, 0], grid_sizes[0, 1], grid_sizes[0, 2])  # [f, h, w]
            first_frame_map = attention_map[0, :, :]  # [h, w]
            
            # sample a point weighted on attention map
            first_frame_map = first_frame_map.view(-1).softmax(dim=0).view(first_frame_map.shape)  # normalize to probabilities

            # take argmax and argmin
            pos_points = self.sample_points(first_frame_map, w, h)  # [2, 2]
            neg_points = self.sample_points(1 - first_frame_map, w, h)  # [2, 2]

            points = torch.cat([pos_points, neg_points], dim=0)  # [2, 2]
            labels = torch.cat([torch.tensor([1]*pos_points.shape[0], dtype=torch.int64), torch.tensor([0]*neg_points.shape[0], dtype=torch.int64)])  # [2], 1 for max, 0 for min

            # points should be [W, H] and not [w, h], normalize by stride because they are used on original_x1 and not on x1
            points = (points * torch.tensor([1.0 * W / w, 1.0 * H / h])).to(torch.int64)  # [2, 2]


            masks = self._compute_subject_mask_given_points(
                latents=x,
                points=points,
                labels=labels
            )

            # masks has shape [F, H, W], downsample the masks with stride to get [f,h,w]
            masks = masks[::stride[0], ::stride[1], ::stride[2]]
            self.masks = torch.Tensor(masks)  # save masks for later use
            return self.masks

    def _compute_subject_mask_given_points(self, latents, points, labels):
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
            masks:  numpy array of shape [f, h, w]
        """
        # decode the latent to get video frames and save them in dir
        videos = self.vae.decode(latents)  # [1, c, f, h, w]
        video = videos[0]  # [c, f, h, w]
        video_dir = str(uuid.uuid4().hex)  # create a unique directory name
        
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # translate points from latent space to video space
        f, h, w = latents[0].shape[-3], latents[0].shape[-2], latents[0].shape[-1]
        F, H, W = video.shape[-3], video.shape[-2], video.shape[-1]
        points = (points * np.array([1.0 * W / w, 1.0 * H / h])).astype(int)    
        
        save_video_tensor_in_dir(video.permute(1,2,3,0), video_dir)
        del videos, video  # free memory
        subject_masks = self._compute_subject_mask_on_decoded_video(video_dir, points, labels)
        delete_video_dir(video_dir)

        # downsample the masks in time (F -> f)
        f_stride = (F-1) // (f-1) if f > 1 else 1
        subject_masks = {
            frame_idx/f_stride: mask 
            for frame_idx, mask in subject_masks.items() 
            if frame_idx % f_stride == 0
        } 
        
        # downsample the masks in space (H,W -> h,w)
        downsampled_masks = {}
        for frame_idx, mask in subject_masks.items():
            downsampled_mask = cv2.resize(mask.astype(np.float32), (latents[0].shape[-1], latents[0].shape[-2]), interpolation=cv2.INTER_LINEAR)
            downsampled_masks[frame_idx] = downsampled_mask

        # convert masks to numpy arrays
        # Convert masks to numpy array of shape [f, h, w]
        masks = np.array([
            downsampled_masks[frame_idx] for frame_idx in sorted(downsampled_masks.keys())
        ], dtype=np.float32)

        return masks
    
    def _compute_subject_mask_on_decoded_video(self, video_dir, points, labels):
        if self.inference_state is not None:
            self.sam2._reset_tracking_results(self.inference_state)
        else:
            self.inference_state = self.sam2.init_state(video_dir)
        
        frame_idx = 0
        obj_id = 1

        _,  out_obj_ids, out_mask_logits = self.sam2.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        subject_masks = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2.propagate_in_video(self.inference_state):
            # Find the mask for our specific object ID
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id == obj_id:
                    subject_masks[out_frame_idx] = np.squeeze((out_mask_logits[i] > 0.0).cpu().numpy())
                    break

        return subject_masks
    
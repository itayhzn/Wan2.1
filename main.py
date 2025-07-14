import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from utils import read_video
from wan.modules.vae import WanVAE

from latent_segmentor import LatentSegmentor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# device = torch.device("cuda")
device = torch.device('cpu')
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    device = 'cuda'
    vae = WanVAE(
            vae_pth=os.path.join('Wan2.1-T2V-1.3B', 'Wan2.1_VAE.pth'),
            device=device)
    
    video_path = "generated/20250709_072402_attn_head_4_subject_mask_t2v-1.3B_832*480_1_1_A_small_brown_dog_playing_with_ADDIT_A_white_kitten.mp4"  # Path to your video file

    video = torch.tensor(np.array(read_video(video_path))) # list of [H, W, C]
    video = video.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [F, H, W, C] -> [1, C, F, H, W]
    video = video.float() / 255.0 # Convert to float and normalize to [0, 1]
    
    latents = vae.encode(video)  # [1, c, f, h, w]
    
    points = np.array([[650, 300], [300, 300]])
    labels = np.array([1, 0])  # 1 for positive point,

    latent_height, latent_width = latents[0].shape[-2], latents[0].shape[-1]
    video_height, video_width = video.shape[-2], video.shape[-1]
    
    points = points * np.array([latent_width / video_width, latent_height / video_height])

    # done with preprocessing, should only use: latents, points, labels from now on

    segmentor = LatentSegmentor(vae=vae, device=device)
    subject_masks = segmentor.compute_subject_mask(latents, points, labels)

    latent_to_visualize = latents[0][:3].cpu().permute(1,2,3,0)  # [f, h, w, 3]
    latent_to_visualize = ((latent_to_visualize - latent_to_visualize.min())/(latent_to_visualize.max() - latent_to_visualize.min())).float().numpy()  # Normalize to [0, 1]

    os.makedirs('sam2_mask_example', exist_ok=True)

    for frame_idx, frame in enumerate(latent_to_visualize):
        # frame dim [C, H, W]
        mask = subject_masks[frame_idx]
        subject = np.multiply(frame, mask[..., np.newaxis])
        background = np.multiply(frame, (1 - mask[..., np.newaxis]))
        
        plt.figure(figsize=(9, 6))
        plt.title(f"subject {frame_idx}")
        plt.imshow(subject)
        plt.savefig(f"sam2_mask_example/subject_{frame_idx:04d}.jpg", bbox_inches='tight', pad_inches=0.1)
        # close
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.title(f"background {frame_idx}")
        plt.imshow(background)
        plt.savefig(f"sam2_mask_example/background_{frame_idx:04d}.jpg", bbox_inches='tight', pad_inches=0.1)
        # close
        plt.close()

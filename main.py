import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# device = torch.device("cuda")
device = torch.device('cpu')
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

from sam2.sam2_video_predictor import SAM2VideoPredictor

from .modules.vae import WanVAE

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
        video.append(frame)
    cap.release()
    return video

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

def save_video_tensor_in_dir(video, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(output_dir, f"{i:04d}.jpg"))

def save_video_from_file_in_dir(video_path, output_dir):
    """
    Save video frames to a directory.
    """
    video = read_video(video_path)
    save_video_tensor_in_dir(video, output_dir)

def delete_video_dir(video_dir):
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)

def compute_subject_mask(video_dir, points, labels):
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    inference_state = predictor.init_state(video_dir)
    
    frame_idx = 0
    obj_id = 1

    _,  out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    subject_masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        subject_masks[out_frame_idx] = np.squeeze((out_mask_logits[obj_id] > 0.0).cpu().numpy())

    return subject_masks

def compute_subject_mask_on_latent(latent, points, labels, vae):
    """
    Run latent through VAE to get video frames.
    Then, use compute_subject_mask to get the subject masks.
    Finally, downsample the masks to match the latent resolution.
    """
    # decode the latent to get video frames and save them in dir
    video = vae.decode(latent)[0]
    video_dir = 'tmp_video_dir'
    save_video_tensor_in_dir(video, video_dir)

    return None

    # translate points from latent space to video space
    latent_height, latent_width = latent.shape[-2], latent.shape[-1]
    video_height, video_width = video.shape[-2], video.shape[-1]
    points = points * np.array([video_width / latent_width, video_height / latent_height])
    
    # compute subject mask
    subject_mask = compute_subject_mask(video_dir, points, labels)
    
    # downsample the masks to match the latent resolution
    downsampled_masks = {}
    for frame_idx, mask in subject_mask.items():
        # downsample mask to match latent resolution
        downsampled_mask = cv2.resize(mask.astype(np.float32), (latent.shape[-1], latent.shape[-2]), interpolation=cv2.INTER_LINEAR)
        downsampled_masks[frame_idx] = downsampled_mask
    
    # delete the video directory
    # delete_video_dir(video_dir)

    return downsampled_masks

def normalize_tensor(tensor):
    """
    Normalize a tensor to the range [0, 1].
    tensor: [f, h, w]
    """
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val == 0:
        return tensor
    t = (tensor - min_val) / (max_val - min_val)
    # change type to float32 for better visualization
    return t.float()

if __name__ == "__main__":
    os.makedirs('tmp_video_dir', exist_ok=True)
    
    device = 'cuda'
    vae = WanVAE(
            vae_pth=os.path.join('Wan2.1-T2V-1.3B', 'Wan2.1_VAE.pth'),
            device=device)
    video_path = "generated/20250709_072402_attn_head_4_subject_mask_t2v-1.3B_832*480_1_1_A_small_brown_dog_playing_with_ADDIT_A_white_kitten.mp4"  # Path to your video file
    
    # subject_masks = compute_subject_mask(video_path, points, labels)

    video = read_video(video_path) # list of frames as numpy arrays

    video_tensor = torch.tensor(np.array(video)).permute(0, 3, 1, 2).to(device) 
    
    latent = vae.encode(video_tensor)[0]  # Encode the video to get the latent representation

    latent = normalize_tensor(latent)  # Normalize the latent for better visualization

    for frame_idx, frame in enumerate(latent):
        plt.figure(figsize=(9, 6))
        plt.title(f"latent {frame_idx}")
        plt.imshow(frame.permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"tmp_video_dir/latent_{frame_idx:04d}.jpg", bbox_inches='tight', pad_inches=0.1)
        # close
        plt.close()

    points = np.array([[650, 300], [300, 300]])
    labels = np.array([1, 0])  # 1 for positive point,

    # compute_subject_mask_on_latent(video_tensor, points, labels, vae)

    # for frame_idx, frame in enumerate(video):
    #     mask = subject_masks[frame_idx]
    #     subject = np.multiply(frame, mask[..., np.newaxis])
    #     background = np.multiply(frame, (1 - mask[..., np.newaxis]))
    #     plt.figure(figsize=(9, 6))
    #     plt.title(f"subject {frame_idx}")
    #     plt.imshow(subject)
    #     plt.savefig(f"tmp_video_dir/subject_{frame_idx:04d}.jpg", bbox_inches='tight', pad_inches=0.1)
    #     # close
    #     plt.close()

    #     plt.figure(figsize=(9, 6))
    #     plt.title(f"background {frame_idx}")
    #     plt.imshow(background)
    #     plt.savefig(f"tmp_video_dir/background_{frame_idx:04d}.jpg", bbox_inches='tight', pad_inches=0.1)
    #     # close
    #     plt.close()

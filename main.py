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

def save_video_in_dir(video_path, output_dir):
    """
    Save video frames to a directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = read_video(video_path)
    for i, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame.save(os.path.join(output_dir, f"{i:04d}.jpg"))
        
if __name__ == "__main__":
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    video_dir = "tmp_video_dir"

    video_path = "generated/20250709_072402_attn_head_4_subject_mask_t2v-1.3B_832*480_1_1_A_small_brown_dog_playing_with_ADDIT_A_white_kitten.mp4"  # Path to your video file
    video_frames = read_video(video_path)
    save_video_in_dir(video_path, video_dir)
    
    inference_state = predictor.init_state(video_dir)

    frame_idx = 0
    obj_id = 1
    points = np.array([[650, 300], [300, 300]])
    labels = np.array([1, 0])  # 1 for positive point,

    _,  out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, f'{frame_idx:04d}.jpg')))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    # save frame to file os.path.join(video_dir, f'{frame_idx}_annotated.jpg')
    plt.savefig(os.path.join(video_dir, f'{frame_idx:04d}_annotated.jpg'), bbox_inches='tight', pad_inches=0.1)
    # close
    plt.close()

    
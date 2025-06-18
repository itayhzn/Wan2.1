import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import rc
from IPython.display import HTML
from matplotlib import rcParams
import seaborn as sns
from textwrap import wrap
from tqdm import tqdm
import pandas as pd
import re

def read_tensors(dir, prompt=None, tensor_names=None):
    tensors = {}
    
    prompts = [ x for x in os.listdir(dir) if x == prompt ] if prompt is not None else [ x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x)) ]
    for prompt in prompts:
        tensors[prompt] = {}       
        for timestep in tqdm(os.listdir(os.path.join(dir, prompt)), desc=f"Loading tensors for prompt {prompt}"):
            t = int(timestep.split('_')[1])  # Extract the timestep from the folder name
            tensors[prompt][t] = {}
            if tensor_names is not None:
                # Filter tensors based on the provided names
                tensor_objs = [tensor_obj for tensor_obj in os.listdir(os.path.join(dir, prompt, timestep)) if tensor_obj[:-3] in tensor_names]
            else:
                tensor_objs = os.listdir(os.path.join(dir, prompt, timestep))
            for tensor_obj in tensor_objs:
                if tensor_obj.endswith('.pt'):
                    key = tensor_obj[:-3]
                    # print(f"Loading tensor {key} for prompt {prompt}, timestep {t}")
                    tensors[prompt][t][key] = torch.load(os.path.join(dir, prompt, timestep, tensor_obj), map_location='cpu')
    return tensors

def add_qk_tensors(tensors, qs=None, ks=None):
    def _compute_qk(q, k):
            # q and k are tensors of shape [B, num_heads, L1, d] and [B, num_heads, L2, d]
            qk = q.transpose(1,2) @ k.transpose(1,2).transpose(-2,-1) # [B, num_heads, L1, L2]
            # divide by sqrt(d) to normalize
            qk = qk / torch.sqrt(qk.shape[-1]) # [B, num_heads, L1, L2]
            # apply softmax to get attention weights
            attn_weights = qk.softmax(dim=-1) # [B, num_heads, L1, L2]
            return attn_weights

    for prompt in tensors.keys():
        for timestep in tensors[prompt].keys():
            qs = qs if qs is not None else [ tensor_obj for tensor_obj in tensors[prompt][timestep] if 'q' in tensor_obj ]
            ks = ks if ks is not None else [ tensor_obj for tensor_obj in tensors[prompt][timestep] if 'k' in tensor_obj ]
            for q in qs:
                for k in ks:
                    tensors[prompt][timestep][f'{q}_{k}'] = _compute_qk(tensors[prompt][timestep][q], tensors[prompt][timestep][k])

    return tensors

# create a grid of subplots with _t rows and _f columns plotting channel 0 of the tensor
def plot_tensor(timesteps, frames, key, channel=0, save_filename=None):
    fig, axs = plt.subplots(len(timesteps), len(frames), figsize=(20, 10.5), dpi=900)
    # fig, axs = plt.subplots(len(timesteps), len(frames), figsize=(20, 18), dpi=900)
    axs = np.atleast_2d(axs)
    
    for i, t in enumerate(timesteps):
        for j, f in enumerate(frames):
            axs[i, j].imshow(get_frame(t, f, key)[:, :, channel], cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    
    for j, f in enumerate(frames):
        axs[0, j].set_title(f"Frame {f}", fontsize=55)
    for i, t in enumerate(timesteps):
        axs[i, 0].set_ylabel(f"t={t}", fontsize=55)
        
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0, hspace=0)
    # plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, format='pdf', bbox_inches='tight')
    plt.show()

# plot an animation of a channel in each timestep
rcParams['animation.embed_limit'] = 2**128
rc('animation', html='jshtml')

def animate_tensor(tensor, title, save_mp4_filename=None, fps=4):
    """
    tensor: [f, h, w]
    title: title of the animation
    save_mp4_filename: if provided, saves the animation as an mp4 file with this name
    fps: frames per second for the animation
    """
    fig = plt.figure(dpi=900)
    ax = fig.add_subplot(111)
    fig.tight_layout()

    ax.set_title(title, fontsize=20)
    im = ax.imshow(tensor[0,:,:], animated=True, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    def updatefig(i):
        ax.set_title(title, fontsize=20)
        t = tensor[i, :, :]
        im.set_array(t)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=tensor.shape[0], interval=50, blit=True)
    if save_mp4_filename:
        print(f'Saving tensor_visualizations/{save_mp4_filename}.mp4')
        ani.save(f'tensor_visualizations/{save_mp4_filename}.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
        
    plt.close(fig)

def animate_channel_over_time(tensor, title, frame_num=0, c=0, save_mp4_filename=None, fps=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title + f" (t={0}, f={frame_num})")
    t = tensor[0, frame_num, :, :, c]
    im = ax.imshow(t, animated=True, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    def updatefig(i):
        ax.set_title(title + f"Timestep {(i+1):02} (frame #{(frame_num+1):02}, channel #{c})", fontsize=20)
        t = tensor[i, frame_num, :, :, c]
        im.set_array(t)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=tensor.shape[0], interval=50, blit=True)
    
    if save_mp4_filename:
        print(f'Saving tensor_visualizations/{save_mp4_filename}_f={frame_num}_c={c}.mp4')
        ani.save(f'tensor_visualizations/{save_mp4_filename}_f={frame_num}_c={c}.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
    
    plt.close(fig)
    return ani

def animate_timestep_video(tensor, title, timestep=0, c=0, save_mp4_filename=None, fps=10):
    fig = plt.figure(dpi=900)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    ax.set_title(title + f"Timestep {(timestep+1):02} (frame #00, channel #{c})", fontsize=20)
    t = tensor[timestep, 0, :, :, :]
    if c == -1:
        t = torch.mean(t, dim=-1)
    else:
        t = t[:, :, c]

    im = ax.imshow(t, animated=True, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    def updatefig(i):
        ax.set_title(title + f"Timestep {(timestep+1):02} (frame #{(i+1):02}, channel #{c})", fontsize=20)
        t = tensor[timestep, i, :, :, c]
        im.set_array(t)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=tensor.shape[1], interval=50, blit=True)
    if save_mp4_filename:
        print(f'Saving tensor_visualizations/{save_mp4_filename}_t={timestep}_c={c}.mp4')
        ani.save(f'tensor_visualizations/{save_mp4_filename}_t={timestep}_c={c}.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
        
    plt.close(fig)
    return ani


if __name__ == "__main__":
    grid_size = [21, 30, 52] # [F, H, W] 

    dir = '/home/ai_center/ai_data/itaytuviah/Wan2.1/tensors/'
    prompt = 'default_t2v-1.3B_832*480_1_1_A_woman_performing_an_intricat_20250612_135951_ADDIT_a_cat'
    tensors = read_tensors(dir, prompt, ['q1', 'k_context1', 'v_context1'])

    # first frame of q1 at timestep 49
    q_frame = tensors[prompt][49]['q1'][0, :, :, :].squeeze(0).permute(1, 0, 2) # [num_heads, F*H*W, head_dim]
    k_frame = tensors[prompt][49]['k_context1'][0, :, :, :].squeeze(0).permute(1, 0, 2) # [num_heads, L, head_dim]

    qk = q_frame @ k_frame.transpose(-2, -1)  # [num_heads, F*H*W, L]

    softmax_qk = F.softmax(qk, dim=-1)  # [num_heads, F*H*W, L]

    # average over heads
    avg_softmax_qk = softmax_qk.mean(dim=0)  # [F*H*W, L]

    # reshape to [F, H, W, L]
    avg_softmax_qk = avg_softmax_qk.reshape(grid_size[0], grid_size[1], grid_size[2], -1)  # [F, H, W, L]
    
    animate_tensor(avg_softmax_qk[:,:,:,3], 'Average Softmax QK')
    
    tensors = add_qk_tensors(tensors, ['q1', 'k1'])
    
    # tensors[prompt][0]['q1'].shape

    # Plotting example
    timesteps = list(tensors['prompt1'].keys())
    frames = [0, 1, 2]  # Example frame indices
    plot_tensor(timesteps, frames, 'qk_tensor', channel=0, save_filename='example_plot.pdf')
    
    # Animation example
    animate_channel_over_time(tensors['prompt1'][timesteps[0]]['qk_tensor'], 'Example Animation', frame_num=0, c=0, save_mp4_filename='example_animation')
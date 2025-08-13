# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as TorchF
from tqdm import tqdm

import numpy as np

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import samwise
from PIL import Image
from utils import save_tensors

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        
        self.samwise_model = samwise.build_samwise_model()
        self.samwise_model.eval().requires_grad_(False)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 edit_mode=False,
                 input_path=None,
                 subject_prompt=None,
                 edit_prompt=None,
                 encoded_params=None
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        subject_mask = None
        ##################
        if edit_mode:
            # 1. read the video from input_path
            vid_folder, frames_list, ext = samwise.extract_frames_from_mp4(input_path)
            video = [ Image.open(os.path.join(vid_folder, frame + ext)).convert('RGB') for frame in frames_list ]
            video = [ torch.tensor(np.array(frame), dtype=torch.float32, device=self.device) for frame in video ]
            video = torch.stack(video, dim=0)  # [F, H, W, C]

            # 2. Update output video parameters to match the input video
            frame_num = video.shape[0]
            size = (video.shape[2], video.shape[1])  # (W, H)

            # 3. Compute the mask for the subject
            mask = samwise.compute_masks(
                self.samwise_model, subject_prompt, vid_folder, frames_list, ext, samwise.get_samwise_args()
            )

            # delete the video folder to save space
            if os.path.exists(vid_folder):
                for frame in frames_list:
                    frame_path = os.path.join(vid_folder, frame + ext)
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                os.rmdir(vid_folder)

        ##################

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        ##################
        if edit_mode:
            # 4. Downsample the mask to match the latent video size
            original_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            subject_mask = original_mask
            # subject_mask = TorchF.interpolate(
            #     original_mask.unsqueeze(0).unsqueeze(0).float(),
            #     size=(target_shape[1], target_shape[2], target_shape[3]),
            #     mode='trilinear',
            #     align_corners=False
            # ).squeeze(0).squeeze(0)

            # 5. save the tensors for debugging
            # save_tensors(f'tensors/{encoded_params}', {
            #     'video': video,
            #     'original_mask': original_mask,
            #     'subject_mask': subject_mask
            # })

            # 6. encode the video. 
            # Current video shape is [F, H, W, C], VAE expects [C, F, H, W]
            anchor_z0 = self.vae.encode(video.permute(3, 0, 1, 2).unsqueeze(0))  # [1, C, F, H, W]
            anchor_z0 = anchor_z0[0] # [C, F, H, W]

        ###################

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device) if input_prompt else None
            context_null = self.text_encoder([n_prompt], self.device) if n_prompt else None
            edit_context = self.text_encoder([edit_prompt], self.device) if edit_prompt else None
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu')) if input_prompt else None
            context_null = self.text_encoder([n_prompt], torch.device('cpu')) if n_prompt else None
            edit_context = self.text_encoder([edit_prompt], torch.device('cpu')) if edit_prompt else None
            context = [t.to(self.device) for t in context] if context else None
            context_null = [t.to(self.device) for t in context_null] if context_null else None
            edit_context = [t.to(self.device) for t in edit_context] if edit_context else None

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len, 'edit_context': edit_context, 'subject_mask': subject_mask, 'edit_mode': edit_mode}
            arg_null = {'context': context_null, 'seq_len': seq_len, 'edit_context': edit_context, 'subject_mask': subject_mask, 'edit_mode': edit_mode}

            anchor_Zt = None
            start_timestep = 0 if edit_mode else 0

            for idx, t in enumerate(tqdm(timesteps)):
                timestep = [t]
                
                if edit_mode:
                    if idx < start_timestep:
                        continue

                    anchor_Zt = sample_scheduler.add_noise(
                        anchor_z0, noise[0], torch.tensor(timestep)) # [C, F, H, W]
                    
                    if idx == start_timestep:
                        latents = [anchor_Zt]

                arg_c['anchor_Zt'] = [anchor_Zt]
                arg_null['anchor_Zt'] = [anchor_Zt]

                latent_model_input = latents

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

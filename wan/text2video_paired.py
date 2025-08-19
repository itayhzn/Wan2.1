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
import uuid

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.pair_model import PairedWanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from utils import save_tensors, save_video_tensor_in_dir, delete_video_dir
from PIL import Image
import samwise

import torch.nn.functional as TF

class PairedWanT2V:

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
        self.model = PairedWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        
        self.samwise_model = samwise.build_samwise_model()

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
                 edit_prompt="",
                 subject_prompt="",
                 encoded_params=None,
                 original_video=None):
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
            edit_prompt (`str`, *optional*, defaults to ""):
                A prompt containing the objects to be added to the video. Will only be added to the second video.

        Returns:
            video1, video2: each are torch.Tensor representing the generated video frames tensors. 
            Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
            The second video will contain the objects specified in `edit_prompt`.
        """
        if original_video is None:
            raise ValueError("original_video must be provided for paired generation.")
        
        if original_video.shape[0] == 3:
            # Convert from [C, F, H, W] to [F, H, W, C]
            video = original_video.permute(1, 2, 3, 0).contiguous()

        original_video_dir = f'original_video_{uuid.uuid4().hex}'
        ext = '.png'
        frames_list = save_video_tensor_in_dir(video, output_dir=original_video_dir, ext=ext)

        # 2. Update output video parameters to match the input video
        frame_num = video.shape[0]
        size = (video.shape[2], video.shape[1])  # (W, H)

        # 3. Compute the mask for the subject
        mask = samwise.compute_masks(
            self.samwise_model, subject_prompt, original_video_dir, frames_list, ext, samwise.get_samwise_args()
        )
        
        original_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        subject_mask = original_mask
        
        # delete the video folder to save space
        delete_video_dir(original_video_dir)

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        subject_mask = TF.interpolate(
            original_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(target_shape[1], target_shape[2], target_shape[3]),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

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
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            edit_context = self.text_encoder([edit_prompt], self.device)
            subject_context = self.text_encoder([subject_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            edit_context = self.text_encoder([edit_prompt], torch.device('cpu'))
            subject_context = self.text_encoder([subject_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            edit_context = [t.to(self.device) for t in edit_context]
            subject_context = [t.to(self.device) for t in subject_context]


        noise1 = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        noise2 = [
            noise1[0].clone().detach()
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
                
                paired_sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                paired_sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                paired_timesteps = paired_sample_scheduler.timesteps

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
            latents1 = noise1
            latents2 = noise2

            print(f'original_mask.shape: {original_mask.shape}, subject_mask.shape: {subject_mask.shape}, latents1.shape: {latents1[0].shape}, latents2.shape: {latents2[0].shape}')

            arg_c = {'context1': context, 'context2': context, 'seq_len': seq_len, 'edit_context': edit_context, 'subject_context': subject_context, 'subject_masks': subject_mask}
            arg_null = {'context1': context_null, 'context2': context_null, 'seq_len': seq_len, 'edit_context': context_null, 'subject_context': context_null, 'subject_masks': subject_mask}

            edit_timesteps = timesteps[7:]


            for idx, t in enumerate(tqdm(timesteps)):
                timestep = [t]

                should_edit = t in edit_timesteps
                arg_c['should_edit'] = should_edit
                arg_null['should_edit'] = should_edit
                
                timestep = torch.stack(timestep)

                self.model.to(self.device)

                arg_c['save_tensors_dir'] = None # f'tensors/{encoded_params}/timestep_{idx}' if encoded_params else None
                
                noise_pred_cond1, noise_pred_cond2 = self.model(
                    latents1, latents2, t=timestep, **arg_c)
                noise_pred_cond1, noise_pred_cond2 = noise_pred_cond1[0], noise_pred_cond2[0]

                noise_pred_uncond1, noise_pred_uncond2 = self.model(
                    latents1, latents2, t=timestep, **arg_null)
                noise_pred_uncond1, noise_pred_uncond2 = noise_pred_uncond1[0], noise_pred_uncond2[0]

                noise_pred1 = noise_pred_uncond1 + guide_scale * (
                    noise_pred_cond1 - noise_pred_uncond1)
                noise_pred2 = noise_pred_uncond2 + guide_scale * (
                    noise_pred_cond2 - noise_pred_uncond2)

                temp_x0_1 = sample_scheduler.step(
                    noise_pred1.unsqueeze(0),
                    t,
                    latents1[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                
                temp_x0_2 = paired_sample_scheduler.step(
                    noise_pred2.unsqueeze(0),
                    t,
                    latents2[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                
                latents1 = [temp_x0_1.squeeze(0)]
                latents2 = [temp_x0_2.squeeze(0)]

                
            x0_1 = latents1
            x0_2 = latents2


            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos1 = self.vae.decode(x0_1)
                videos2 = self.vae.decode(x0_2)

        del noise1, noise2, latents1, latents2
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos1[0] if self.rank == 0 else None, videos2[0] if self.rank == 0 else None

    
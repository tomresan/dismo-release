# Adapted from 
# https://github.com/THUDM/CogKit/blob/main/src/cogkit/finetune/diffusion/models/cogvideo/cogvideox_i2v/lora_trainer.py

import os
import logging
from functools import partial
from typing import Any, Sequence, Callable, Dict
from jaxtyping import Float, UInt8
from pathlib import Path
from copy import deepcopy
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from peft import LoraConfig, inject_adapter_in_model
from einops import rearrange

from dismo.model import MotionExtractor, MappingNetwork
from dismo.video_model_finetuning.lora_layers import DataProvider, ConditionalLinear
from dismo.data import apply_geometric_transformations


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def resized_center_crop(x: torch.Tensor, size: int | list) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    ratio = size[0] / size[1]
    return apply_geometric_transformations(
        frames=x,
        size=size, 
        min_aspect_ratio=ratio,
        max_aspect_ratio=ratio,
        # min_aspect_ratio=1,
        # max_aspect_ratio=1,
    )


def get_captions(x, kwargs: dict) -> Sequence[str]:
    txt = []
    for i in range(x.shape[0]):
        if 'txt' in kwargs and kwargs['txt'][i] is not None:
            txt.append(kwargs['txt'][i])
        else:
            txt.append('')
    return txt


class CogVideoXMotionAdapter(nn.Module):
    DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

    def __init__(
        self, 
        model_path: str | Path,
        motion_extractor_params: Dict[str, Any],
        # motion_extractor_ckpt_path: str | Path,
        d_cond_lora: int = 1024,
        text_conditioning: bool = True,
        weight_dtype = torch.bfloat16,
        enable_gradient_checkpointing: bool = True,
        train_resolution: list[int] = [25, 480, 720],
        vae_slicing: bool = False,
        vae_tiling: bool = False,
    ) -> None:
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.motion_extractor = MotionExtractor(**motion_extractor_params)
        # self.motion_extractor.load_state_dict(torch.load(motion_extractor_ckpt_path, map_location='cpu'))
        self.model_path = model_path
        self.d_cond_lora = d_cond_lora
        self.text_conditioning = text_conditioning
        self.train_resolution = train_resolution
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.is_i2v = 'I2V' in str(model_path)
        self.weight_dtype = weight_dtype
        self.enable_slicing = vae_slicing
        self.enable_tiling = vae_tiling
        self.low_vram = False
        self.data_provider = DataProvider()
        self.load_components()
        self.prepare_models()
        self.prepare_trainable_parameters()
        self.motion_embs_per_latent = int(self.vae.config.temporal_compression_ratio * (self.transformer.config.patch_size_t or 1))
        
        d_motion = self.motion_extractor.head[3].scale.shape[0]
        self.gru = nn.GRUCell(input_size=d_motion, hidden_size=d_cond_lora)
        self.init_gru_state = nn.Parameter(torch.zeros(d_cond_lora))
        self.mapping = MappingNetwork(n_layers=1, d_model=d_cond_lora, d_ff=3*d_cond_lora, dropout=0)
        self.final_latent_cond_padding = nn.Parameter(torch.zeros(1, 1, d_cond_lora))

        self.encode_video = torch.compile(self.encode_video, fullgraph=True, dynamic=False)
        self._encode_motion_compiled = torch.compile(self._encode_motion_compiled, fullgraph=True, dynamic=False)
        if self.text_conditioning:
            self.text_encoder.forward = torch.compile(self.text_encoder.forward, fullgraph=True, dynamic=False)
        if not self.enable_gradient_checkpointing:
            self.transformer.forward = torch.compile(self.transformer.forward, fullgraph=True, dynamic=False)

    def load_components(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        dtype = self.weight_dtype
        model_path = str(self.model_path)

        ### tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", token=os.environ.get("HF_TOKEN"))

        ### text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            token=os.environ.get("HF_TOKEN")
        )

        ### transformer
        if not self.low_vram:
            self.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                token=os.environ.get("HF_TOKEN")
            )
        else:
            self.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=dtype,
                token=os.environ.get("HF_TOKEN")
            )

        ### vae
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
            token=os.environ.get("HF_TOKEN")
        )

        ### scheduler
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler",
            token=os.environ.get("HF_TOKEN")
        )

        if not self.text_conditioning:
            empty_text_embeddings = self.encode_text("")
            self.register_buffer('empty_text_embeddings', empty_text_embeddings)
            del self.text_encoder

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        text_embeddings = self.text_encoder(prompt_token_ids.to(self.text_encoder.device)).last_hidden_state[0]

        # shape of text_embeddings: [seq_len, hidden_size]
        assert text_embeddings.ndim == 2
        return text_embeddings
    
    def encode_motion(self, x: torch.Tensor):
        B = x.shape[0]
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = F.interpolate(
            x,
            size=(self.motion_extractor.train_resolution[1], self.motion_extractor.train_resolution[2]),
            mode='bilinear',
            antialias=True,
        )
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B)
        with torch.no_grad():
            embs = self.motion_extractor.forward_sliding(x)
        return self._encode_motion_compiled(embs)

    def _encode_motion_compiled(self, embs: torch.Tensor):
        first_hidden = self.gru(embs[:, 0], self.init_gru_state.expand(embs.shape[0], -1))
        next_embs = rearrange(embs[:, 1:], "b (n p) c -> p (b n) c", p=self.motion_embs_per_latent)
        next_hiddens = self.init_gru_state.expand(next_embs.shape[1], -1)
        for next_emb in next_embs:
            next_hiddens = self.gru(next_emb, next_hiddens)
        next_hiddens = rearrange(next_hiddens, "(b n) c -> b n c", b=embs.shape[0])
        hiddens = torch.cat([first_hidden[:, None], next_hiddens], dim=1)
        embs = self.mapping(hiddens)
        embs = torch.cat([embs, self.final_latent_cond_padding.expand(embs.shape[0], -1, -1)], dim=1)
        return embs.to(dtype=self.weight_dtype)
    
    def forward(self, x, **kwargs) -> dict[str, Any]:
        if self.text_conditioning:
            txt = get_captions(x, kwargs)
            text_embeddings = torch.stack([self.encode_text(t) for t in txt])
        else:
            text_embeddings = self.empty_text_embeddings.expand(x.shape[0], -1, -1)

        motion_embs = self.encode_motion(x)
        latent = self.encode_video(x.permute(0, 4, 1, 2, 3)) # channels first

        device = latent.device

        patch_size_t = self.transformer.config.patch_size_t

        if patch_size_t is not None:
            # Copy the first frame ncopy times to match patch_size_t
            ncopy = latent.shape[2] % patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.scheduler.add_noise(latent, noise, timesteps)
        latent_img_noisy = latent_noisy

        # Prepare I2V
        if self.is_i2v: # B H W C
            images = x[:, 0].permute(0, 3, 1, 2)  # to [B,C,H,W]

            # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
            images = images.unsqueeze(2)

            # Add noise to images
            image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=device)
            image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
            noisy_images = (
                images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
            )
            with torch.no_grad():
                image_latent_dist = self.vae.encode(noisy_images.to(dtype=self.vae.dtype)).latent_dist
                image_latents = image_latent_dist.sample() * self.vae.config.scaling_factor
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
            assert (latent.shape[0], *latent.shape[2:]) == (
                image_latents.shape[0],
                *image_latents.shape[2:],
            )

            # Padding image_latents to the same frame number as latent
            padding_shape = (
                latent.shape[0],
                latent.shape[1] - 1,
                *latent.shape[2:],
            )
            latent_padding = image_latents.new_zeros(padding_shape)
            image_latents = torch.cat([image_latents, latent_padding], dim=1)

            # Concatenate latent and image_latents in the channel dimension
            latent_img_noisy = torch.cat([latent_img_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=self.transformer.config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=device,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )

        self.data_provider.set(cond_lora=motion_embs, start_idx=text_embeddings.shape[1])
        predicted_noise = self.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=text_embeddings,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        # self.data_provider.reset()

        # Denoise
        latent_pred = self.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean(
            (weights * (latent_pred - latent) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        loss = loss.mean()

        return loss, {}

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    def prepare_models(self) -> None:
        if self.vae is not None:
            if self.enable_slicing:
                self.vae.enable_slicing()
            if self.enable_tiling:
                self.vae.enable_tiling()

    def get_lora_config(self) -> LoraConfig:
        lora_config =  LoraConfig(
            r=128,
            lora_alpha=64,
            init_lora_weights=True,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        )
        def layer_factory(*args, **kwargs):
            return ConditionalLinear(*args, **kwargs, d_cond=self.d_cond_lora, data_provider=self.data_provider)
        lora_config._register_custom_module({ torch.nn.Linear: layer_factory })
        return lora_config

    def prepare_trainable_parameters(self) -> None:
        if self.text_conditioning:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
            self.text_encoder.train = lambda x: x
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.train = lambda x: x
        self.motion_extractor.requires_grad_(False)
        self.motion_extractor.eval()
        self.motion_extractor.train = lambda x: x
        self.transformer.requires_grad_(False)
        lora_config = self.get_lora_config()
        inject_adapter_in_model(lora_config, self.transformer, adapter_name="default")
        if self.enable_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
    
    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def sample(
        self, 
        prompts: Sequence[str] | None = None, 
        images: Float[torch.Tensor, "b h w c"] = None, # (-1, 1)
        motion_videos: Float[torch.Tensor, "b t h w c"] = None, # (-1, 1)
        width: int | None = None, 
        height: int | None = None, 
        num_frames: int | None = None, 
        text_guidance_scale: float = 6,
        n_steps: int = 50, 
        generator: Sequence[torch.Generator] | torch.Generator | None = None,
        negative_prompt: str | None = None,
        progress_bar: bool = False,
    ) -> UInt8[torch.Tensor, "b t h w c"]: # (0, 255)

        num_frames = num_frames or self.train_resolution[0]
        height = height or self.train_resolution[1]
        width = width or self.train_resolution[2]

        # text
        if not self.text_conditioning and prompts is not None:
            logging.warning("Text conditioning is disabled")

        if self.text_conditioning:
            if prompts is None:
                prompts = [""] * len(motion_videos)
                text_guidance_scale = 1
            text_embeddings = torch.stack([self.encode_text(p) for p in prompts])
            negative_text_embeddings = self.encode_text(negative_prompt or self.DEFAULT_NEGATIVE_PROMPT).expand(len(motion_videos), -1, -1)
        else:
            text_embeddings = self.empty_text_embeddings.expand(len(motion_videos), -1, -1)
            negative_text_embeddings = self.empty_text_embeddings.expand(len(motion_videos), -1, -1)
            text_guidance_scale = 1

        # motion
        motion_embeddings = (
            self.encode_motion(motion_videos) 
            if motion_videos is not None else 
            [None] * len(text_embeddings)
        )

        # source images
        if images is None:
            images = [None] * len(motion_videos)

        # sampling
        videos = []
        for i, (text_embs, neg_text_embs, motion_embs, image) in enumerate(zip(text_embeddings, negative_text_embeddings, motion_embeddings, images)):
            pipeline_inputs = {
                "prompt_embeds": text_embs[None, :],
                "negative_prompt_embeds": neg_text_embs[None, :],
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": n_steps,
                "guidance_scale": text_guidance_scale,
                "generator": generator[0] if isinstance(generator, (list, tuple)) else generator,
                "return_dict": False,
            }

            if self.is_i2v and image is not None:
                image = image.permute(2, 0, 1).add(1).div(2).unsqueeze(0) # (b c h w) [-1, 1] -> [0, 1]
                if image.shape[-2] != height or image.shape[-1] != width:
                    image = image.mul(255)
                    image = resized_center_crop(image, size=(width, height))
                    image = image.float().div(255)
                pipeline_inputs["image"] = image

            if motion_embs is not None:
                motion_embs = motion_embs.unsqueeze(0)
                if text_guidance_scale > 1:
                    motion_embs = torch.cat([motion_embs, motion_embs], dim=0)
                self.data_provider.set(cond_lora=motion_embs, start_idx=text_embs.shape[0])

            pipeline_cls = (
                CogVideoXImageToVideoPipeline 
                if images is not None and self.is_i2v else 
                CogVideoXPipeline
            )
            pipeline = pipeline_cls(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder if self.text_conditioning else None,
                vae=self.vae,
                transformer=self.transformer,
                scheduler=deepcopy(self.scheduler),
            )
            pipeline.set_progress_bar_config(disable=not progress_bar)

            video = pipeline(**pipeline_inputs)[0]
            video = torch.from_numpy(np.stack([np.array(f) for f in video]))[0]
            videos.append(video)
            self.data_provider.reset()
            free_memory()
        
        videos = torch.stack(videos)
        return videos
    


CogVideoXMotionAdapter_5B_TI2V_Large = partial(
    CogVideoXMotionAdapter, 
    model_path="THUDM/CogVideoX-5B-I2V",
    motion_extractor_params=dict(
        width=1024,
        depth=20,
        d_head=64,
        d_motion=128,
        frame_encoder_params=dict(
            model_version="dinov2_vitl14_reg",
            gradient_last_blocks=2,
        ),
        max_delta_time=4,
        train_resolution=[8, 256, 256],
    ),
    d_cond_lora=1024,
    text_conditioning=True,
    train_resolution=[25, 480, 720],
)

from abc import ABC
import math
from typing import List, Sequence, Callable, Dict, Any
from collections.abc import Sequence
import logging
import os
import random
import time
from pathlib import Path
from functools import partial
from jaxtyping import Float, UInt8

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import _functional_tensor as F_t
from torch.utils.data import IterableDataset

from torchcodec.decoders import VideoDecoder

import webdataset as wds



# ================================================================================================
# Data augmentation utilities
# ================================================================================================

# Copied and adapted from torchvision to support non-uniform scaling
def get_inverse_affine_matrix(
    center: List[float],
    angle: float,
    translate: List[float],
    scale: List[float],
    shear: List[float],
    inverted: bool = True,
) -> List[float]:
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        matrix = [d / scale[0], -b / scale[0], 0.0, -c / scale[1], a / scale[1], 0.0]
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a * scale[1], b * scale[0], 0.0, c * scale[1], d * scale[0], 0.0]
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def affine(img, angle, translate, scale, shear, center, size, padding_mode="zeros"):
    b, c, h, w = img.shape
    matrix = get_inverse_affine_matrix(center, angle, [float(t) for t in translate], scale, shear)
    theta = torch.tensor(matrix, dtype=img.dtype, device=img.device).view(1, 2, 3)
    grid = F_t._gen_affine_grid(theta, w=w, h=h, ow=size[0], oh=size[1]).expand(b, -1, -1, -1)
    return F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )


def _sample_param(rng_spec, default, generator=None, n=1):
    """
    Draws * n * values uniformly from (lo, hi); returns default(s) if spec is None.
    """
    if rng_spec is None:
        return (default,) * n if n > 1 else default
    lo, hi = rng_spec
    draw = lambda: float(torch.rand((), generator=generator) * (hi - lo) + lo)
    return tuple(draw() for _ in range(n)) if n > 1 else draw()


# Adapted from torchvision.transforms.RandomResizedCrop
def random_crop(scale_spec, ratio_spec, img_height: int, img_width: int, generator: torch.Generator = None):
    area = img_height * img_width
    log_ratio0, log_ratio1 = math.log(ratio_spec[0]), math.log(ratio_spec[1])

    scales = (torch.rand(10, generator=generator) * (scale_spec[1]-scale_spec[0]) + scale_spec[0]).tolist()
    ratios = (torch.rand(10, generator=generator) * (log_ratio1-log_ratio0) + log_ratio0).tolist()

    for scale, log_r in zip(scales, ratios):
        target_area = area * scale
        aspect_ratio = math.exp(log_r)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= img_width and 0 < h <= img_height:
            i = int(torch.randint(0, img_height - h + 1, size=(), generator=generator))
            j = int(torch.randint(0, img_width - w + 1, size=(), generator=generator))
            return (j, i, w, h)

    # fallback
    in_ratio = img_width / img_height
    if in_ratio < ratio_spec[0]:
        w = img_width
        h = int(round(w / ratio_spec[0]))
    elif in_ratio > ratio_spec[1]:
        h = img_height
        w = int(round(h * ratio_spec[1]))
    else:
        w = img_width
        h = img_height
    i = (img_height - h) // 2
    j = (img_width - w) // 2
    return (j, i, w, h)


def apply_geometric_transformations(
    frames: Float[torch.Tensor, "b c h w"], 
    generator: torch.Generator | None = None, 
    **params
) -> Float[torch.Tensor, "b c h w"]:
    generator = generator or torch.Generator().manual_seed(int(time.time() * 1e3) % 2**32)

    # get frame dimensions etc.
    t, c, h_orig, w_orig = frames.shape
    x, y, w, h = 0, 0, w_orig, h_orig
    r = w / h # aspect ratio

    # optionally crop frames to fit aspect ratio constraints
    r_max = params.get("max_aspect_ratio", None) # e.g. 4/3
    r_min = params.get("min_aspect_ratio", None) # e.g. 3/4
    if r_max is not None and r > r_max:
        w_new = int(round(h * r_max))
        x = x + int((w - w_new) / 2)
        w = w_new
    elif r_min is not None and r < r_min:
        h_new = int(round(w / r_min))
        y = y + int((h - h_new) / 2)
        h = h_new

    # optionally perform a random crop (similar to torchvision.transforms.RandomResizedCrop)
    if params.get("random_crop", False):
        x_add, y_add, w, h = random_crop(
            scale_spec=params.get("scale"),
            ratio_spec=params.get("aspect_ratio"),
            img_width=w,
            img_height=h,
            generator=generator,
        )
        x = x + x_add
        y = y + y_add

    # determine output size
    size_out = params.get("size", (w, h))
    if isinstance(size_out, (int, float)):
        w_out, h_out = size_out, size_out
    else:
        w_out, h_out = size_out
    
    # determine affine transformation parameters
    center = (
        x + (w * 0.5) - (w_orig * 0.5),
        y + (h * 0.5) - (h_orig * 0.5),
    )
    translate = (-center[0], -center[1])
    scale = (w_out / w, h_out / h)

    if not params.get("random_crop", False):
        s = _sample_param(params.get("scale"), 1.0, generator=generator)
        tx, ty = _sample_param(params.get("translate"), 0.0, n=2, generator=generator)
        scale = (scale[0] * s, scale[1] * s)
        translate = (translate[0] + (tx * w), translate[1] + (ty * h))

    angle = _sample_param(params.get("angle"), 0.0, generator=generator)
    shear = _sample_param(params.get("shear"), 0.0, generator=generator)

    # apply affine transformation
    frames = affine(
        frames,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[shear, 0], # skew in x-direction only
        center=center,
        size=(w_out, h_out),
        padding_mode=params.get("padding_mode", "border"),
    )
    return frames


def apply_photometric_transformations(
    frames: Float[torch.Tensor, "b c h w"], 
    generator: torch.Generator | None = None, 
    **params
) -> Float[torch.Tensor, "b c h w"]:
    generator = generator or torch.Generator().manual_seed(int(time.time() * 1e3) % 2**32)

    OPS = dict(
        brightness=TF.adjust_brightness,
        contrast=TF.adjust_contrast,
        saturation=TF.adjust_saturation,
        hue=TF.adjust_hue,
    )

    # Build lambdas that capture the sampled value
    transforms = [
        (lambda x, v=_sample_param(params[k], None, generator=generator), f=f: f(x, v)) 
        for k, f in OPS.items() if k in params
    ]

    # Shuffle once with the same generator, then apply
    for idx in torch.randperm(len(transforms), generator=generator):
        frames = transforms[idx](frames)

    return frames


def apply_transformations(
    frames: UInt8[torch.Tensor, "b c h w"], 
    params_photometric: Dict[str, Any] | None = None, 
    params_geometric: Dict[str, Any] | None = None,
) -> UInt8[torch.Tensor, "b c h w"]:
    frames = frames.float().div(255)
    if params_geometric is not None:
        frames = apply_geometric_transformations(frames, **params_geometric)
    if params_photometric is not None:
        frames = apply_photometric_transformations(frames, **params_photometric)
    # cast back to uint8 to reduce memory footprint during data shuffling
    frames = frames.mul(255).round().byte()
    return frames


def get_transform(params_photometric=None, params_geometric=None) -> Callable[[torch.Tensor], torch.Tensor]:
    return partial(apply_transformations, params_photometric=params_photometric, params_geometric=params_geometric)



# ================================================================================================
# Base video loader class
# ================================================================================================

def identity(x):
    return x


def unbatch(source):
    for batch in source:
        yield from batch


class ResampledShardLists(IterableDataset):
    def __init__(self, tar_paths, n_repeats=1, shuffle=False, seed=1337, continue_from_step=0):
        super().__init__()
        if isinstance(tar_paths, str):
            self.tar_paths = wds.shardlists.expand_urls(tar_paths)
        else:
            self.tar_paths = list(tar_paths)
        self.tar_paths = sorted(self.tar_paths)
        self.n_repeats = n_repeats
        self.shuffle = shuffle
        if self.shuffle:
            self.rng = random.Random(seed)
        self.from_step = continue_from_step

    def __iter__(self):
        count = 0
        for _ in range(self.n_repeats):
            tar_paths = self.tar_paths.copy()
            if self.shuffle:
                self.rng.shuffle(tar_paths)
            for tar_path in tar_paths:
                if count >= self.from_step:
                    yield dict(url=str(tar_path))
                count += 1


class BaseVideoLoader(ABC):
    VIDEO_EXTENSIONS = ["mp4", "avi", "webm", "mpg"]
    VIDEO_NAME_KEY = "video_name"
    VIDEO_URL_KEY = "video_url"

    def __init__(
        self,
        data_paths: str | Sequence[str],
        batch_size: int = 1,
        num_workers: int = 0,
        video_extension: str | None = None,
        repeats: int = 1,
        epoch_size: int | None = None,
        shardshuffle: bool = False,
        shuffle: int = 0,
        partial: bool = True,
        pin_memory: bool = False,
        node_splitting: bool = True,
        deterministic: bool = False,
        seed: int = 1337,
    ):
        super().__init__()
        if isinstance(data_paths, str):
            data_paths = list(Path(data_paths).rglob(f"*.tar"))
        elif isinstance(data_paths, Sequence):
            assert all(isinstance(p, str) for p in data_paths)
        else:
            raise Exception("`data_paths` must either be a directory or a sequence of tar paths.")
        self.data_paths = data_paths
        self.video_extension = video_extension
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.partial = partial
        self.n_repeats = repeats
        self.shardshuffle = shardshuffle
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.deterministic = deterministic
        self.node_splitting = node_splitting
        self.n_workers = int(num_workers)
        self.seed = seed
        self.reset()
    
    def yield_samples(self, video_decoder: VideoDecoder, **kwargs):
        return
    
    def construct_batch(self, samples: list):
        return samples

    def reset(self):
        self.global_rng = random.Random(self.seed)
        rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
        seed_list = [self.seed, rank]
        if not self.deterministic:
            seed_list.extend([os.getpid(), time.time_ns(), os.urandom(4)])
        self.local_rng = random.Random(wds.utils.make_seed(*seed_list))
        self.loader = None
        self.world_size = world_size
        self.rank = rank
        logging.info(f"Reset video loader (rank={rank})")

    def __iter__(self):
        pipeline = []
        pipeline.append(
            ResampledShardLists(
                self.data_paths,
                n_repeats=self.n_repeats,
                shuffle=self.shardshuffle,
                seed=int.from_bytes(self.global_rng.randbytes(4)),
            )
        )
        if self.node_splitting:
            pipeline.append(wds.split_by_node)
        pipeline.append(wds.split_by_worker)
        pipeline.append(wds.tarfile_to_samples(handler=wds.warn_and_continue))
        pipeline.append(wds.pipelinefilter(self.process_shard_entries)())

        dataset = wds.DataPipeline(*pipeline)

        loader = wds.WebLoader(
            dataset,
            batch_size=8, # performs better than 1
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2 if self.n_workers > 0 else None,
            drop_last=False,
            collate_fn=identity,
        )
        loader.append(wds.pipelinefilter(unbatch)())
        if self.shuffle > 0:
            loader.append(wds.shuffle(self.shuffle, initial=self.shuffle, rng=self.local_rng))
        loader.append(wds.pipelinefilter(self.batch)())

        yield from loader

    def process_shard_entries(self, source):
        for entry in source:
            video_name = ""
            try:
                video_name = entry["__key__"]

                # extract encoded video bytes from entry
                if self.video_extension is not None:
                    assert (
                        self.video_extension in entry
                    ), f"Video extension {self.video_extension} not found for {video_name}. Skipping."
                    enc_video = entry.pop(self.video_extension)
                else:
                    enc_video = None
                    ks = tuple(entry.keys())
                    for e in self.VIDEO_EXTENSIONS:
                        for k in ks:
                            if k.endswith(e):
                                assert (
                                    enc_video is None
                                ), "Multiple video entries found in sample. Skipping due to ambiguity."
                                enc_video = entry.pop(k)
                    assert enc_video is not None, "No video entry found in sample. Skipping."

                # construct video decoder
                video_decoder = VideoDecoder(enc_video, num_ffmpeg_threads=0)

                # yield samples from video
                yield from self.yield_samples(video_decoder, **entry)
            
            except Exception as e:
                logging.warning(f"{video_name}: {e}")

    def batch(self, source):
        buffer = []
        for sample in source:
            if sample is None:
                continue
            buffer.append(sample)
            if len(buffer) >= self.batch_size:
                batch = self.construct_batch(buffer)
                if batch is not None:
                    yield batch
                buffer = []
        if self.partial and len(buffer) > 0:
            batch = self.construct_batch(buffer)
            yield batch



# ================================================================================================
# General video loader used for video model finetuning
# ================================================================================================

class VideoLoader(BaseVideoLoader):
    def __init__(
        self, 
        *args,
        clip_length: int | float | None = None,
        clips_per_video: int | None = None,
        clip_shift: bool = False,
        fps: float | None = None,
        transform = None, 
        additional_keys = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_length = clip_length
        self.clips_per_video = clips_per_video
        self.clip_shift = clip_shift
        self.fps = fps
        self.transform = transform
        self.additional_keys = additional_keys
        if self.additional_keys is not None:
            self.wds_decoder = wds.autodecode.Decoder(handlers=[])

    def yield_samples(self, video_decoder: VideoDecoder, **kwargs):
        if self.additional_keys is not None:
            kwargs = { k: v for k, v in kwargs.items() if k in self.additional_keys }
            kwargs = self.wds_decoder(kwargs)

        stride = 1 if self.fps is None else int(round(video_decoder.metadata.average_fps / self.fps))
        n_video_frames = video_decoder.metadata.num_frames
        clip_length = self.clip_length or (n_video_frames - 1) // stride + 1
        clip_range = (clip_length - 1) * stride + 1

        if n_video_frames < clip_range:
            return

        n_clips = self.clips_per_video or int(n_video_frames / clip_range)
        for clip_idx in range(n_clips):
            try:
                if self.clips_per_video is not None:
                    offset = self.local_rng.random() * (n_video_frames - clip_range)
                else:
                    offset = clip_idx * clip_range
                    if self.clip_shift:
                        offset += (self.local_rng.random() - 0.5) * clip_range
                offset = min(max(int(round(offset)), 0), n_video_frames - clip_range)
                frame_idcs = [offset + (i * stride) for i in range(clip_length)]
                frame_batch = video_decoder.get_frames_at(frame_idcs)
                frames = frame_batch.data
                if self.transform is not None:
                    frames = self.transform(frames)
                yield { "x": frames, **kwargs }
            except Exception as e:
                logging.warning(f"Error extracting clip: {e}")

    def construct_batch(self, samples):
        batch = {}
        if self.clip_length is not None:
            batch["x"] = torch.stack([s["x"] for s in samples]).permute(0, 1, 3, 4, 2).float().div(127.5).sub(1)
        else:
            batch["x"] = [s["x"].permute(0, 2, 3, 1).float().div(127.5).sub(1) for s in samples]
        if self.additional_keys is not None:
            for k in self.additional_keys:
                batch[k] = [s.get(k, None) for s in samples]
        return batch
    


# ================================================================================================
# Video loader specialized for DisMo pre-training
# ================================================================================================

class DismoVideoLoader(BaseVideoLoader):
    def __init__(
        self,
        *args,
        clip_length: int,
        clips_per_video: int | None = None,
        clip_shift: bool = False,
        fps: float | None = None,
        max_delta_time_frames: int | None = None,
        delta_time_distribution: dict | float | int | None = {'type': 'gamma', 'concentration': 3.0, 'rate': 12.0},
        motion_transform=None,
        content_transform=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_length = clip_length
        self.clips_per_video = clips_per_video
        self.clip_shift = clip_shift
        self.fps = fps
        self.max_delta_time_frames = max_delta_time_frames
        dist_type = delta_time_distribution.pop('type', 'gamma')
        if dist_type == 'gamma':
            self.delta_time_distribution = torch.distributions.gamma.Gamma(**delta_time_distribution) # in seconds
        elif dist_type == 'constant':
            self.delta_time_distribution = None
            self.n_delta_time_frames = delta_time_distribution['num_frames']
        self.motion_transform = motion_transform
        self.content_transform = content_transform

    def yield_samples(self, video_decoder: VideoDecoder, **kwargs):
        n_video_frames = video_decoder.metadata.num_frames        
        video_fps = video_decoder.metadata.average_fps
        stride = 1 if self.fps is None else int(round(video_fps / self.fps))
        n_clip_frames = (self.clip_length - 1) * stride + 1
        n_clips = self.clips_per_video or int(n_video_frames / n_clip_frames)

        if n_video_frames < n_clip_frames:
            return

        for clip_idx in range(n_clips):
            try:
                if self.clips_per_video is not None:
                    offset = self.local_rng.random() * (n_video_frames - n_clip_frames)
                else:
                    offset = clip_idx * n_clip_frames
                    if self.clip_shift:
                        offset += (self.local_rng.random() - 0.5) * n_clip_frames
                offset = min(max(int(round(offset)), 0), n_video_frames - n_clip_frames)
                motion_idcs = [offset + (i * stride) for i in range(self.clip_length)]

                if self.delta_time_distribution is not None:
                    distances = torch.arange((self.max_delta_time_frames * stride) + 1).float().div(video_fps)
                    probs = self.delta_time_distribution.log_prob(distances).exp()
                    probs = probs.div(probs.sum()) # normalize
                    target_offsets = torch.multinomial(probs, num_samples=self.clip_length - self.max_delta_time_frames, replacement=True).tolist()
                    target_idcs = [motion_idcs[i] + target_offsets[i] for i in range(self.clip_length - self.max_delta_time_frames)]
                else:
                    distances = torch.tensor([stride / video_fps * self.n_delta_time_frames])
                    target_offsets = [0] * (self.clip_length - self.max_delta_time_frames)
                    target_idcs = motion_idcs[self.n_delta_time_frames:]

                motion_frames = video_decoder.get_frames_at(motion_idcs).data
                source_frames = motion_frames[:(self.clip_length - self.max_delta_time_frames)]
                target_frames = video_decoder.get_frames_at(target_idcs).data

                motion_frames = self.motion_transform(motion_frames)
                source_frames, target_frames = torch.stack([self.content_transform(torch.stack([source_frames[i], target_frames[i]])) for i in range(len(source_frames))], dim=1).unbind(0)

                yield dict(
                    motion_frames=motion_frames,
                    source_frames=source_frames,
                    target_frames=target_frames,
                    delta_times=distances[target_offsets].mul(video_fps / stride),
                )
            except Exception as e:
                logging.warning(f"Error extracting clip: {e}")

    def construct_batch(self, samples):
        batch = {}
        batch["motion_frames"] = torch.stack([s["motion_frames"] for s in samples]).permute(0, 1, 3, 4, 2).float().div(127.5).sub(1) # b t h w c
        batch["source_frames"] = torch.stack([s["source_frames"] for s in samples]).permute(0, 1, 3, 4, 2).float().div(127.5).sub(1) # b p h w c
        batch["target_frames"] = torch.stack([s["target_frames"] for s in samples]).permute(0, 1, 3, 4, 2).float().div(127.5).sub(1) # b p h w c
        batch["delta_times"] = torch.stack([s["delta_times"] for s in samples]) # b p
        return batch

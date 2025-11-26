# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Thomas Ressler-Antal et al., CompVis @ LMU Munich

import os
import sys
from pathlib import Path
import logging
import random

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm.auto import tqdm

from dismo.model import DisMo, DisMo_Large
from dismo.data import DismoVideoLoader


def endless_iter(iterable):
    while True:
        yield from iterable


# Main training entry point
# Add arguments here to make them configurable via CLI
def train(
    # General
    out_dir="output",
    load_checkpoint: str | None = None,
    checkpoint_freq: int = 10_000,
    max_steps: int = 1_000_000,
    warmup_steps: int = 10_000,
    # Data
    data_paths: str = "data",
    # Training
    local_batch_size: int = 32,
    lr: float = 5e-5,
    clip_grad_norm: float = 1.0,
    # Misc
    compile: bool = False,
    enable_wandb: bool = True,
):
    train_params = locals()
    # Output & logging setup
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log"),
        ],
    )

    # Distributed init & handling of single-GPU case
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_type = "cuda"
        device = torch.device(f"{device_type}:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Running distributed. Local rank: {local_rank}, World size: {world_size}")

        rank0logger = logging.getLogger(__name__)
        if rank != 0:
            rank0logger.disabled = True
        barrier = dist.barrier
    else:
        rank = 0
        device_type = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        logger.info(f"Running non-distributed on {device_type}")

        rank0logger = logger
        barrier = lambda: None

    # WandB setup
    if enable_wandb and rank == 0:
        import wandb

        wandb.init(
            project="dismo",
            config=train_params | {"global_batch_size": local_batch_size * world_size},
            dir=out_dir,
        )

    # Checkpoint loading pt1
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        start_step = checkpoint["step"]
        rank0logger.info(f"Loaded checkpoint from {load_checkpoint} @ step {start_step}.")
    else:
        checkpoint = None
        start_step = 0

    # Seeding
    seed = 42 + rank + start_step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Important setup stuff
    # If you want to change anything about what you train, you'll likely want to do it here and add it as a parameter to train()
    model = DisMo_Large(compile=compile).to(device)
    data = DismoVideoLoader(
        data_paths=data_paths, 
        batch_size=local_batch_size, 
        shuffle=1000, 
        shardshuffle=True,
        repeats=sys.maxsize,
        num_workers=16,
        pin_memory=True,
        partial=False,
        seed=start_step+42,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), fused=compile)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=1e-8),
        ],
        milestones=[warmup_steps],
    )
    rank0logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M"
        f" ({sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M trainable)"
    )

    # Checkpoint loading pt2: actually loading state
    if load_checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    # DDP wrapping
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    barrier()

    # Training loop
    if rank == 0:
        logger.info("Starting training...")
    for i, batch in enumerate(
        pbar := tqdm(endless_iter(data.make_loader()), desc="Training", initial=start_step, disable=rank != 0)
    ):
        try:
            batch = { k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items() }
            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss, metrics = model(**batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()

            avg_loss = (
                (dist_nn.all_reduce(loss.detach().clone(), op=dist.ReduceOp.SUM) / world_size)
                if is_distributed
                else loss.detach()
            )

            metrics = {
                k: (
                    dist_nn.all_reduce(v.detach(), op=dist.ReduceOp.SUM) / world_size if is_distributed else v.detach()
                ).item()
                for k, v in metrics.items()
            }
            train_meta = {
                "loss": avg_loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            } | metrics

            pbar.set_postfix(train_meta)
            if enable_wandb and rank == 0:
                wandb.log({f"train/{k}": v for k, v in train_meta.items()}, step=start_step + i)

            done = max_steps is not None and (start_step + i) >= max_steps
            if done:
                rank0logger.info(f"Reached max steps: {start_step + i} >= {max_steps}. Stopping training...")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping training...")
            done = True
        if done or (i % checkpoint_freq == 0 and rank == 0) and i > 0:
            # Save checkpoint
            checkpoint = {
                "model": (model.module if is_distributed else model).state_dict(),  # type: ignore
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": start_step + i,
            }
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, ckpt_dir / f"checkpoint_{start_step + i:07}.pt")
            rank0logger.info(f"Saved checkpoint at step {start_step + i}.")

        if done:
            break
    barrier()
    rank0logger.info("Training stopped.")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    # By launching with fire, all arguments become specifyable via the CLI
    # e.g. python train.py --data_paths /path/to/data --local_batch_size 32
    try:
        import fire

        fire.Fire(train)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
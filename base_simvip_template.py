# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchinfo import summary
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.cuda.amp import autocast
import numpy as np
import time
import csv
import os
from tqdm import tqdm
import zarr
from omegaconf import OmegaConf
import wandb
import argparse
import xarray as xr

import sys
from hydra import main, initialize, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

# Append model path dynamically (could also be in config)
if os.path.exists('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src'):
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src')
else:
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
import simvip_model
import chatgpt_data_loaders 

# Precompute filters once (Sobel + Laplacian)
class GradientFilters(nn.Module):
    def __init__(self, channels, device, dtype):
        super().__init__()
        self.channels = channels
        self.register_buffer("weight_x", self.make_kernel([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], channels, device, dtype))
        self.register_buffer("weight_y", self.make_kernel([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], channels, device, dtype))
        self.register_buffer("weight_lap", self.make_kernel([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], channels, device, dtype))
    def make_kernel(self, kernel_2d, channels, device, dtype):
        kernel = torch.tensor(kernel_2d, dtype=dtype, device=device)
        return kernel.expand(channels, 1, 3, 3).contiguous()
    def compute_first_order(self, x):
        dx = F.conv2d(x, self.weight_x, padding=1, groups=self.channels)
        dy = F.conv2d(x, self.weight_y, padding=1, groups=self.channels)
        return dx, dy
    def compute_second_order(self, x):
        lap = F.conv2d(x, self.weight_lap, padding=1, groups=self.channels)
        return lap

# Compute loss with optional masking and per-data-point normalization
def compute_loss(model,x,y,mode,alpha0,alpha1,alpha2,filters,masked_loss=False,):
    model.train() if mode == 'train' else model.eval()
    # Get model's dtype and device
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    # Move and cast x, y accordingly
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)
    B, T, C, H, W = y.shape
    y_hat = model(x)
    # Reshape for 2D filtering: [B*T, C, H, W]
    # This is because GradientFilters (which uses F.conv2d) expects input.shape == [N, C, H, W]
    y2d = y.reshape(B*T, C, H, W)
    yhat2d = y_hat.reshape(B*T, C, H, W)
    if masked_loss:
        mask2d = (y2d != 0)
    else:
        mask2d = None
    
    # ==== MSE loss ====
    if masked_loss:
        mse_raw = (y2d - yhat2d) ** 2
        mse_loss = (mse_raw * mask2d).sum() / mask2d.sum().clamp_min(1.0)
    else:
        mse_loss = F.mse_loss(y2d, yhat2d)
        
    # ==== First-order gradients ====
    dx_y, dy_y = filters.compute_first_order(y2d)
    dx_yhat, dy_yhat = filters.compute_first_order(yhat2d)
    grad_x_loss = (dx_y - dx_yhat).pow(2)
    grad_y_loss = (dy_y - dy_yhat).pow(2)
    if masked_loss:
        # Mask center pixels (e.g., pad=1 conv ⇒ shrink mask by 1)
        central_mask_x = mask2d[..., 1:] * mask2d[..., :-1]
        central_mask_y = mask2d[..., 1:, :] * mask2d[..., :-1, :]
        central_mask_x = central_mask_x.reshape(B*T, C, H, W - 1)
        central_mask_y = central_mask_y.reshape(B*T, C, H - 1, W)
        grad_x_loss = (grad_x_loss[..., 1:] * central_mask_x).sum() / central_mask_x.sum().clamp_min(1.0)
        grad_y_loss = (grad_y_loss[..., 1:, :] * central_mask_y).sum() / central_mask_y.sum().clamp_min(1.0)
    else:
        grad_x_loss = grad_x_loss.mean()
        grad_y_loss = grad_y_loss.mean()
    grad_loss = grad_x_loss + grad_y_loss

    # ==== Second-order: Laplacian ====
    lap_y = filters.compute_second_order(y2d)
    lap_yhat = filters.compute_second_order(yhat2d)
    lap_loss = (lap_y - lap_yhat).pow(2)
    if masked_loss:
        # Center 3-frame mask
        central_mask = (
            mask2d[..., 1:-1, 1:-1]
            * mask2d[..., :-2, 1:-1]
            * mask2d[..., 2:, 1:-1]
            * mask2d[..., 1:-1, :-2]
            * mask2d[..., 1:-1, 2:]
        )
        central_mask = central_mask.reshape(B*T, C, H - 2, W - 2)
        lap_loss = (lap_loss[..., 1:-1, 1:-1] * central_mask).sum() / central_mask.sum().clamp_min(1.0)
    else:
        lap_loss = lap_loss.mean()

    total_loss = alpha0 * mse_loss + alpha1 * grad_loss + alpha2 * lap_loss
    return total_loss
    

class WandbLogger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        self.header_written = False
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def log(self, data: dict):
        if wandb.run:
            wandb.log(data)
        if self.log_path:
            write_header = not os.path.exists(self.log_path) or not self.header_written
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if write_header:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(data)

def train_model(
    model,
    device,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    lr,
    alpha0,
    alpha1,
    alpha2,
    masked_loss,
    checkpoint_dir="checkpoints",
    wandb_config=None,
    hydra_cfg=None,
    dry_run=False,
):
    import wandb  # Safe here for local + cloud hybrid logging

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.flash_sdp_enabled = True
    scaler = torch.cuda.amp.GradScaler()
    use_amp = device.type == "cuda"

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    os.makedirs("logs", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float("inf")
    start_epoch = 0
    global_step = 0
    filters = None

    # Checkpoint loading...
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"Found existing checkpoint. Loading: {best_ckpt_path}")
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith(f"{model_name}_checkpoint_") and f.endswith(".pt")])
        if checkpoint_files:
            latest_ckpt = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resuming training from epoch {start_epoch}")

    # W&B Initialization
    wandb_logger = WandbLogger(log_path=f"logs/{model_name}_wandb.csv") if wandb_config else None

    if wandb_config:
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config.get("model_name", model_name),
            config={"lr": lr, "num_epochs": num_epochs, "alpha0": alpha0, "alpha1": alpha1, "alpha2": alpha2}
        )
        wandb.run.log_code(".")
        if hydra_cfg:
            wandb.config.update(OmegaConf.to_container(hydra_cfg, resolve=True), allow_val_change=True)

    # Example dry-run logging (optional)
    log_path = f"logs/{model_name}_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "epoch_time_sec"])

    # Profiler warmup...
    ################################################################
    # Profiler
    if hydra_cfg.training.get("enable_profiler", True):
        prof_schedule = schedule(
            wait=1,  # warm-up
            warmup=1,
            active=3,  # active profiling
            repeat=1
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_logs/detailed")
        ) as prof:
            for _ in range(5):
                print("Running detailed dry-run for profiling...")
                torch.cuda.synchronize()
                start = time.time()
                x, y = next(iter(train_loader))
                x, y = x.to(device, non_blocking=True, dtype=torch.float32), y.to(device, non_blocking=True, dtype=torch.float32)
                C = y.shape[-3]
                filters = GradientFilters(C, device=model_device, dtype=model_dtype)
                torch.cuda.synchronize()
                print("Dataloader time:", time.time() - start)
                start = time.time()
                with autocast(enabled=use_amp), record_function("training_step"):
                    loss = compute_loss(model, x, y, 'train', alpha0, alpha1, alpha2, filters, masked_loss)
                scaler.scale(loss).backward()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                print("GPU time:", time.time() - start)
                prof.step()
        # Print top time-consuming operations to console
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        print(f"Dry-run complete. GPU mem: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
    if dry_run:
        return model
    ################################################################

    for epoch in range(start_epoch, num_epochs):
        # Inside training loop
        model.train()
        train_losses = []
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            if not filters:
                C = y.shape[-3]
                filters = GradientFilters(C, device=model_device, dtype=model_dtype)
            if filters.channels != y.shape[-3]:
                C = y.shape[-3]
                filters = GradientFilters(C, device=model_device, dtype=model_dtype)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = compute_loss(model, x, y, 'train', alpha0, alpha1, alpha2, filters, masked_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            log_data = {
                "train/step_loss": loss.item(),
                "gpu_mem_allocated": torch.cuda.memory_allocated(device) / (1024**3),
                "global_step": global_step
            }
            if wandb_logger:
                wandb_logger.log(log_data)
            global_step += 1
        train_loss = np.mean(train_losses)

        # Validation...
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                val_loss = compute_loss(model, x, y, 'val', alpha0, alpha1, alpha2, filters, masked_loss)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        print(f"[{epoch+1}] Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", round(epoch_time, 2)])
        if not np.isfinite(val_loss):
            print("Skipping save due to non-finite val_loss.")
            continue
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_best.pt"))
            print(f"Best model updated @ step {global_step} (Val Loss: {best_val_loss:.4f})")
        # Save checkpoint
        ckpt_filename = f"{model_name}_checkpoint_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, os.path.join(checkpoint_dir, ckpt_filename))
        if wandb_logger:
            wandb_logger.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "epoch_time_sec": epoch_time,
            })
    # Test
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            test_loss = compute_loss(model, x, y, 'val', alpha0, alpha1, alpha2, filters, masked_loss)
            test_losses.append(test_loss.item())
    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss:.4f}")
    if wandb_logger:
        wandb_logger.log({"test_loss": test_loss})
        wandb.finish()

    return model


# Main entry point
@main(config_path="conf", config_name="config")
def run(cfg):
    # Save config used in the run directory
    os.makedirs(".", exist_ok=True)
    OmegaConf.save(cfg, "config_used.yaml")
    print(f"Config saved at: {os.getcwd()}/config_used.yaml")

    # Configure
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cpus = torch.get_num_threads() if torch.cuda.is_available() else 0
    multiprocessing = torch.cuda.is_available()
    print(f"Using device: {device}")
    print(f"Using n_cpus: {n_cpus}")
    print(f"Multiprocessing: {multiprocessing}")
    print(f"Training model: {cfg['model_variant']}")
    
    # Prepare standards for transforms
    standards = {
                "mean_ssh":  cfg.data["mean_ssh"], "std_ssh": cfg.data["std_ssh"],
                "mean_sst": cfg.data["mean_sst"], "std_sst": cfg.data["std_sst"]
                }
    # Load patch coordinates
    if ".npy" in cfg.data.patch_coords_file:
        patch_coords = np.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    elif ".zarr" in cfg.data.patch_coords_file:
        patch_coords = zarr.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    elif ".nc" in cfg.data.patch_coords_file:
        patch_coords = xr.open_dataarray(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    else:
        print(f"Wrong datatype detected in {cfg.data.patch_coords_file}, expected one of [.npy, .zarr, .nc]")
    # Load dataset
    full_dataset = ConcatDataset([
        chatgpt_data_loaders.llc4320_dataset(
            cfg.data.dataset_path, t, cfg.data.N_t, patch_coords,
            cfg.data.infields, cfg.data.outfields, cfg.data.in_mask_list, cfg.data.out_mask_list,
            cfg.data.in_transform_list, cfg.data.out_transform_list, standards=standards,
            multiprocessing=multiprocessing, return_masks=cfg.data.return_masks
        ) for t in np.arange(cfg.data.timesteps_range[0],cfg.data.timesteps_range[1],cfg.data.timesteps_range[2],)
    ])
    train_len = int(0.7 * len(full_dataset))
    val_len = int(0.2 * len(full_dataset))
    test_len = len(full_dataset) - train_len - val_len
    print(f"Training dataset length: {train_len}")
    print(f"Validation dataset length: {val_len}")
    print(f"Test dataset length: {test_len}")
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    # Instatiate data loaders
    def worker_init_fn(worker_id):
        _ = torch.utils.data.get_worker_info()

    def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))
    #worker_info = torch.utils.data.get_worker_info()


    data_loader_kwargs = {"shuffle":True,
                          "batch_size":batch_size=cfg.data.batch_size,
                          "num_workers":cfg.training.get("workers",8),
                          "worker_init_fn":worker_init_fn,
                          "persistent_workers":cfg.training.get("persistent_workers",True),#multiprocessing,
                          "pin_memory":True,
                          "prefetch_factor":4,
                         #"multiprocessing_context":'spawn'
                         }
    train_loader = DataLoader(train_set,**data_loader_kwargs)
    val_loader = DataLoader(val_set, **data_loader_kwargs)
    test_loader = DataLoader(test_set, **data_loader_kwargs)
    
    # Instantiate model
    in_shape = (cfg.model.Number_timesteps, len(cfg.data.infields), 128, 128)
    base_model = simvip_model.SimVP_Model_no_skip_sst(in_shape=in_shape, **cfg.model)
    base_model = base_model.to(device, non_blocking=True).to(torch.float32) # Put model on GPU
    base_model = torch.compile(base_model, mode="default")
    print("Model is on device:", next(base_model.parameters()).device) # Verify
    
    # Train model
    train_model(
        model=base_model,
        device=device,
        model_name=cfg["model_variant"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        alpha0=cfg.model.alpha0,
        alpha1=cfg.model.alpha1,
        alpha2=cfg.model.alpha2,
        masked_loss=cfg.model.masked_loss,
        wandb_config=cfg.wandb,
        hydra_cfg=cfg,  # Log the entire config to wandb
        dry_run=cfg.training.dry_run,
    )
    
if __name__ == "__main__":
    run()

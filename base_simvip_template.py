# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
import time
import csv
import os
from tqdm import tqdm
import zarr
from omegaconf import OmegaConf
import wandb

import sys
from hydra import main, initialize, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

# Append model path dynamically (could also be in config)
sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
import simvip_model
import data_loaders 

# Define Sobel gradients as before
class SobelGradients5D(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer('weight_x', sobel_x.view(1,1,3,3))
        self.register_buffer('weight_y', sobel_y.view(1,1,3,3))

    def forward(self, x):
        B,C,T,H,W = x.shape
        x_reshaped = x.permute(0,2,1,3,4).reshape(B*T*C,1,H,W)
        grad_x = F.conv2d(x_reshaped, self.weight_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.weight_y, padding=1)
        grad_x = grad_x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        grad_y = grad_y.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        return grad_x, grad_y


# Compute loss with optional masking and per-data-point normalization
def compute_loss(model,batch,mode,alpha0,alpha1,alpha2,device,masked_loss=False,):
    mse = lambda a, b: (a - b) ** 2
    def grad2d(u):
        dx = u[..., :, 1:] - u[..., :, :-1]
        dy = u[..., 1:, :] - u[..., :-1, :]
        return dx, dy
    def grad2d_second(u):
        dxx = u[..., :, 2:] - 2 * u[..., :, 1:-1] + u[..., :, :-2]
        dyy = u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]
        return dxx, dyy
    if mode == 'train':
        model.train()
    else:
        model.eval()
        
    if len(batch) == 2:
        x, y = batch
    else:
        x, y, _ = batch
    x = x.to(device).float()
    y = y.to(device).float()
    y_hat = model(x)
    
    if masked_loss:
        # mask == 1 where y is not zero
        mask = (y != 0).float()
        # Main loss only over unmasked elements
        per_element_loss = mse(y, y_hat) * mask
        loss = per_element_loss.sum() / mask.sum().clamp_min(1.0)
        # Gradients: mask central pixels to avoid border artifacts
        central_mask_x = (mask[..., :, 1:] * mask[..., :, :-1])
        central_mask_y = (mask[..., 1:, :] * mask[..., :-1, :])
        dx_y, dy_y = grad2d(y)
        dx_yh, dy_yh = grad2d(y_hat)
        grad_loss_x = mse(dx_y, dx_yh) * central_mask_x
        grad_loss_y = mse(dy_y, dy_yh) * central_mask_y
        loss_grad = (
            grad_loss_x.sum() / central_mask_x.sum().clamp_min(1.0)
            + grad_loss_y.sum() / central_mask_y.sum().clamp_min(1.0)
        )
        # Second gradients: mask central pixels
        central_mask_xx = (mask[..., :, 2:] * mask[..., :, 1:-1] * mask[..., :, :-2])
        central_mask_yy = (mask[..., 2:, :] * mask[..., 1:-1, :] * mask[..., :-2, :])
        dxx_y, dyy_y = grad2d_second(y)
        dxx_yh, dyy_yh = grad2d_second(y_hat)
        grad2_loss_x = mse(dxx_y, dxx_yh) * central_mask_xx
        grad2_loss_y = mse(dyy_y, dyy_yh) * central_mask_yy
        loss_grad2 = (
            grad2_loss_x.sum() / central_mask_xx.sum().clamp_min(1.0)
            + grad2_loss_y.sum() / central_mask_yy.sum().clamp_min(1.0)
        )
    else:
        # Unmasked case: treat all pixels as valid
        valid_count = torch.numel(y)
        per_element_loss = mse(y, y_hat)
        loss = per_element_loss.sum() / valid_count
        dx_y, dy_y = grad2d(y)
        dx_yh, dy_yh = grad2d(y_hat)
        grad_loss_x = mse(dx_y, dx_yh)
        grad_loss_y = mse(dy_y, dy_yh)

        # For gradients, the counts are smaller because of the shifts:
        count_dx = dx_y.numel()
        count_dy = dy_y.numel()
        loss_grad = (
            grad_loss_x.sum() / count_dx
            + grad_loss_y.sum() / count_dy
        )
        dxx_y, dyy_y = grad2d_second(y)
        dxx_yh, dyy_yh = grad2d_second(y_hat)
        count_dxx = dxx_y.numel()
        count_dyy = dyy_y.numel()
        grad2_loss_x = mse(dxx_y, dxx_yh)
        grad2_loss_y = mse(dyy_y, dyy_yh)
        loss_grad2 = (
            grad2_loss_x.sum() / count_dxx
            + grad2_loss_y.sum() / count_dyy
        )

    total_loss = alpha0 * loss + alpha1 * loss_grad + alpha2 * loss_grad2
    return total_loss
    

# Training loop
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
    wandb_config=None
):
    import wandb

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,500], gamma=0.1)

    # Initialize wandb
    if wandb_config:
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config.get("run_name", model_name),
            config={
                "lr": lr,
                "num_epochs": num_epochs,
                "alpha0": alpha0,
                "alpha1": alpha1,
                "alpha2": alpha2,
            }
        )
        wandb.run.log_code(".")
        
    log_path = f"logs/{model_name}_log.csv"
    if not os.path.exists(log_path):
        with open(log_path,'w',newline='') as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","epoch_time_sec"])
    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)):
            batch = batch
            optimizer.zero_grad()
            loss = compute_loss(model, batch, 'train', alpha0, alpha1, alpha2, device, masked_loss)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # Wandb log per batch
            if global_step % 10 == 0:
                if wandb_config:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
            global_step += 1

        scheduler.step()
        train_loss = np.mean(train_losses)
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                val_loss = compute_loss(model, batch, 'val', alpha0, alpha1, alpha2, device, masked_loss)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        epoch_time = time.time() - start_time
        print(f"[{epoch+1}] Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        # Save checkpoint every epoch
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        ckpt_filename = f"{model_name}_checkpoint_{epoch+1}.pt"
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, ckpt_filename))
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_best.pt"))
            print(f"✅ Best model updated (Val Loss: {best_val_loss:.4f})")
        # Append CSV log
        with open(log_path,'a',newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, round(epoch_time,2)])
        # Wandb log summary stats per epoch
        if wandb_config:
            wandb.log({
                "epoch": epoch+1,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "epoch_time_sec": epoch_time,
            })
    # Test
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            test_loss = compute_loss(model, batch, 'test', alpha0, alpha1, alpha2, device, masked_loss)
            test_losses.append(test_loss.item())
    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss:.4f}")

    if wandb_config:
        wandb.log({"test_loss": test_loss})
        wandb.finish()

    return model

# Main entry point
@main(config_path="conf", config_name="config")
def run(cfg):
    # Configure
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_cpus = torch.get_num_threads() if torch.cuda.is_available() else 0
    multiprocessing = torch.cuda.is_available()

    # Prepare standards for transforms
    standards = {
                "mean_ssh":  cfg.data["mean_ssh"], "std_ssh": cfg.data["std_ssh"],
                "mean_sst": cfg.data["mean_sst"], "std_sst": cfg.data["std_sst"]
                }
    # Load dataset
    patch_coords = zarr.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    full_dataset = ConcatDataset([
        data_loaders.llc4320_dataset(
            cfg.data.dataset_path, t, cfg.data.N_t, patch_coords,
            cfg.data.infields, cfg.data.outfields, cfg.data.in_mask_list, cfg.data.out_mask_list,
            cfg.data.in_transform_list, cfg.data.out_transform_list, standards=standards,
            multiprocessing=multiprocessing, return_masks=cfg.data.return_masks
        ) for t in cfg.data.timesteps_range
    ])
    train_len = int(0.7 * len(full_dataset))
    val_len = int(0.2 * len(full_dataset))
    test_len = len(full_dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])
    
    # Instatiate data loaders
    def worker_init_fn(worker_id):
        _ = torch.utils.data.get_worker_info()
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn=worker_init_fn, persistent_workers=multiprocessing)
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn=worker_init_fn, persistent_workers=multiprocessing)
    test_loader = DataLoader(test_set, batch_size=cfg.data.batch_size, shuffle=False, num_workers=n_cpus, worker_init_fn=worker_init_fn, persistent_workers=multiprocessing)
    
    # Instantiate model
    in_shape = (cfg.model.Number_timesteps, len(cfg.data.infields), 128, 128)
    base_model = simvip_model.SimVP_Model_no_skip_sst(in_shape=in_shape, **cfg.model)
    
    # Train model
    train_model(
        model=base_model,
        device=device,
        model_name=cfg.training.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.model.lr,
        alpha0=cfg.model.alpha0,
        alpha1=cfg.model.alpha1,
        alpha2=cfg.model.alpha2,
        masked_loss=cfg.model.masked_loss,
        wandb_config=cfg.wandb
    )

if __name__ == "__main__":
    run()

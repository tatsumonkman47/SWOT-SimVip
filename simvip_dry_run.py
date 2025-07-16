# dry_run.py
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchinfo import summary
import zarr
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize, compose
from pathlib import Path
import sys
import os
import subprocess
import re

# Append model path dynamically (could also be in config)
if os.path.exists('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src'):
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src')
else:
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
import simvip_model
import data_loaders 
from datetime import datetime


def save_model_summary(
    model,
    input_shape,
    out_dir="summaries",
    gpu_mem_gb=None,
    safety_factor=0.8
):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, f"model_summary_{timestamp}.out")

    # Run torchinfo summary and get the string
    s = summary(
        model,
        input_size=input_shape,
        dtypes=[torch.float32],
        verbose=2,
        col_names=["input_size", "output_size", "num_params"]
    )
    summary_str = str(s)

    # Extract memory estimates from the string using regex
    input_mb = forward_mb = params_mb = total_mb = None
    pattern = re.compile(r"Input size \(MB\): ([\d\.]+)")
    m = pattern.search(summary_str)
    if m:
        input_mb = float(m.group(1))
    pattern = re.compile(r"Forward/backward pass size \(MB\): ([\d\.]+)")
    m = pattern.search(summary_str)
    if m:
        forward_mb = float(m.group(1))
    pattern = re.compile(r"Params size \(MB\): ([\d\.]+)")
    m = pattern.search(summary_str)
    if m:
        params_mb = float(m.group(1))
    pattern = re.compile(r"Estimated Total Size \(MB\): ([\d\.]+)")
    m = pattern.search(summary_str)
    if m:
        total_mb = float(m.group(1))

    if None in (input_mb, forward_mb, params_mb, total_mb):
        raise ValueError("Failed to parse memory usage from torchinfo summary.")

    # Per-sample activations
    batch_size = input_shape[0]
    per_sample_mb = forward_mb / batch_size + input_mb / batch_size

    # GPU memory
    if gpu_mem_gb is not None:
        total_gpu_mem_mb = gpu_mem_gb * 1024
        gpu_source = f"Prescribed GPU memory ({gpu_mem_gb} GB)"
    else:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            total_gpu_mem_mb = int(output.strip().split("\n")[0])
            gpu_source = "Auto-detected GPU memory (nvidia-smi)"
        except Exception:
            total_gpu_mem_mb = 0
            gpu_source = "Unavailable (nvidia-smi failed)"

    # Estimate max batch size
    estimated_batch = None
    if total_gpu_mem_mb > 0:
        avail_mb = total_gpu_mem_mb * safety_factor
        estimated_batch = int((avail_mb - params_mb) / per_sample_mb)

    # Compose extra info
    extra_str = (
        f"\n\n=== Batch Size Estimation ===\n"
        f"Per-sample activation memory: ~{per_sample_mb:.2f} MB\n"
        f"Parameter memory: {params_mb:.2f} MB\n"
    )
    if total_gpu_mem_mb > 0:
        extra_str += (
            f"GPU Memory Source: {gpu_source}\n"
            f"Total GPU Memory: {total_gpu_mem_mb/1024:.2f} GB\n"
            f"Safety Factor: {safety_factor*100:.0f}% usable\n"
            f"Estimated Max Batch Size: {estimated_batch}\n"
        )
    else:
        extra_str += (
            "⚠️ Could not estimate GPU memory automatically. Provide gpu_mem_gb argument to enable estimation.\n"
        )

    # Write output
    with open(filename, "w") as f:
        f.write(summary_str)
        f.write(extra_str)

    print(f"Model summary + estimation saved to {filename}")

    return {
        "per_sample_mb": per_sample_mb,
        "param_mb": params_mb,
        "total_gpu_mem_mb": total_gpu_mem_mb,
        "estimated_batch_size": estimated_batch
    }
    


def dry_run():
    # Force CPU
    device = torch.device("cpu")
    torch.set_default_device(device)
    print(f"💻 Using device: {device}")

    # Load config
    with initialize(config_path="conf"):
        cfg = compose(config_name="000_base_config")
    print("Config loaded")

    # Dummy dataset coords
    if ".npy" in cfg.data.patch_coords_file:
        patch_coords = np.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    elif ".zarr" in cfg.data.patch_coords_file:
        patch_coords = zarr.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    elif ".nc" in cfg.data.patch_coords_file:
        patch_coords = zarr.load(f"{cfg.data.dataset_path}/{cfg.data.patch_coords_file}")
    else:
        print(f"Wrong datatype detected in {cfg.data.patch_coords_file}, expected one of [.npy, .zarr, .nc]")
    # Transforms
    standards = {
        "mean_ssh": cfg.data["mean_ssh"], "std_ssh": cfg.data["std_ssh"],
        "mean_sst": cfg.data["mean_sst"], "std_sst": cfg.data["std_sst"]
    }

    # Build full dataset
    full_dataset = ConcatDataset([
        data_loaders.llc4320_dataset(
            cfg.data.dataset_path, t, cfg.data.N_t, patch_coords,
            cfg.data.infields, cfg.data.outfields,
            cfg.data.in_mask_list, cfg.data.out_mask_list,
            cfg.data.in_transform_list, cfg.data.out_transform_list,
            standards=standards,
            multiprocessing=False,
            return_masks=cfg.data.return_masks
        ) for t in cfg.data.timesteps_range
    ])
    print(f"Full dataset created with {len(full_dataset)} samples")

    # Create dataloader
    loader = DataLoader(full_dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    x, y = batch[:2]
    print(f"Loaded one batch: x.shape = {x.shape}, y.shape = {y.shape}")

    # Build model
    in_shape = (cfg.model.Number_timesteps, len(cfg.data.infields), 128, 128)
    model = simvip_model.SimVP_Model_no_skip_sst(in_shape=in_shape, **cfg.model)
    model.to(device)
    model.eval()
    print("Model instantiated")

    # Save model summary
    save_model_summary(model, input_shape=x.shape)

    # Forward pass
    with torch.no_grad():
        y_hat = model(x.float())
    print(f"🎯 Forward pass successful. Output shape: {y_hat.shape}")

    # Simple assertion check
    assert y.shape == y_hat.shape, "Output shape mismatch!"

    print("Dry run completed successfully!")

if __name__ == "__main__":
    dry_run()
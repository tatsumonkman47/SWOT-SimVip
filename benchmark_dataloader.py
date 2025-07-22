import argparse
import os
import time
import csv
from datetime import datetime
import multiprocessing
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import cProfile, pstats
import numpy as np
import zarr
from hydra import initialize, compose

import sys, os
sys.path.append("/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src")
import chatgpt_data_loaders

def get_args():
    p = argparse.ArgumentParser(description="Benchmark PyTorch DataLoader grid search")
    p.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4], help='List of num_workers to test')
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[4, 8, 16], help='Batch sizes to test')
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--persistent_workers', action='store_true')
    p.add_argument('--collate_fn', action='store_true')
    p.add_argument('--context', type=str, default=None)
    return p.parse_args()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)

def fast_collate(batch):
    invar = torch.stack([b[0] for b in batch])
    outvar = torch.stack([b[1] for b in batch])
    return invar, outvar

def build_dataloader(cfg, args, batch_size, num_workers):
    DATASET_PATH = cfg.data["dataset_path"]
    if os.path.exists('/home/tm3076/projects/NYU_SWOT_project/'):
        patch_coords = zarr.load(f'{DATASET_PATH}/x_y_coordinates_noland.zarr')
    else:
        patch_coords = np.load(f'{DATASET_PATH}/zarred_UVSST_x_y_coordinates_noland_nonan.npy')
    ds = ConcatDataset([
        chatgpt_data_loaders.llc4320_dataset(
            DATASET_PATH, t0, cfg.data["N_t"], patch_coords,
            cfg.data["infields"], cfg.data["outfields"],
            cfg.data["in_mask_list"], cfg.data["out_mask_list"],
            cfg.data["in_transform_list"], cfg.data["out_transform_list"],
            standards={
                "mean_ssh": cfg.data["mean_ssh"], "std_ssh": cfg.data["std_ssh"],
                "mean_sst": cfg.data["mean_sst"], "std_sst": cfg.data["std_sst"]
            },
            cloud_rho=cfg.data["cloud_rho"],
            return_metadata=False, time_loading=False
        ) for t0 in range(30, 360, 5)
    ])
    train_len = int(0.7*len(ds))
    val_len = int(0.2*len(ds))
    train_ds, _, _ = random_split(ds, [train_len, val_len, len(ds)-train_len-val_len])
    kw = dict(
        dataset=train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        worker_init_fn=worker_init_fn, shuffle=True
    )
    if args.collate_fn:
        kw['collate_fn'] = fast_collate
    if args.context:
        kw['multiprocessing_context'] = args.context
    return DataLoader(**kw)

def time_one(loader, n_iter=10):
    # Skip first batch (warmup)
    it = iter(loader)
    next(it)
    tic = time.time()
    for _ in range(n_iter):
        next(it)
    return (time.time() - tic) / n_iter

def main():
    args = get_args()
    with initialize(config_path="conf"):
        cfg = compose(config_name="002_C2_config_07rho")
    cfg.data["dataset_path"] = "/home/tm3076/scratch/pytorch_learning_tiles" if os.path.exists('/home/tm3076/projects/') else "/home.ufs/tm3076/swot_SUM03/pytorch_learning_tiles"
    os.makedirs("benchmarks", exist_ok=True)

    rows = []
    for w in args.workers:
        for b in args.batch_sizes:
            dl = build_dataloader(cfg, args, b, w)
            t = time_one(dl, n_iter=20)
            rows.append({
                "num_workers": w, "batch_size": b,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.persistent_workers,
                "collate_fn": args.collate_fn,
                "context": args.context or "",
                "time_per_batch": t
            })
            print(f"Workers={w}, BS={b}, time={t:.4f}s")

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"benchmarks/benchmark_{timestamp}.csv"
    keys = list(rows[0].keys())
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved results to {fname}")

    # Plot
    import pandas as pd
    df = pd.DataFrame(rows)
    for w in sorted(df.num_workers.unique()):
        sub = df[df.num_workers == w]
        plt.plot(sub.batch_size, sub.time_per_batch, '-o', label=f"wk{w}")
    plt.xlabel("Batch Size")
    plt.ylabel("Time per batch (s)")
    plt.title("DataLoader benchmark")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outpng = fname.replace(".csv", ".png")
    plt.savefig(outpng)
    print(f"Plot saved to {outpng}")
    plt.show()

if __name__ == "__main__":
    """ Example usage:
    python benchmark_dataloader.py --workers 0 2 4 --batch_sizes 4 8 16 --pin_memory --persistent_workers --collate_fn --context fork
    """
    main()

"""
Generalization Test Script (SAFE VERSION)

Tests pre-trained models (trained on 64-bit parity) on longer sequences:
- 128-bit
- 256-bit
- 512-bit
"""

# -------------------------
# HARD GUARDS (MUST BE FIRST)
# -------------------------
import os
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_RESUME"] = "never"
os.environ["WANDB_DISABLE_CODE"] = "true"

# -------------------------
# Imports
# -------------------------
import argparse
import json
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple
from collections import defaultdict
import ast

from data.custom_datasets import ParityDataset
from models.ctm import ContinuousThoughtMachine
from models.lagged_ctm import LaggedContinuousThoughtMachine
from utils.losses import parity_loss


# -------------------------
# SAFE LOCAL SEED (NO W&B)
# -------------------------
def set_seed_local(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lagged_checkpoint", type=str, required=True)
    p.add_argument("--baseline_checkpoint", type=str, required=True)
    p.add_argument("--sequence_lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--n_test_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="validation/generalization_outputs")
    p.add_argument("--wandb_project", type=str, default="lagged-ctm-validation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------------------------
# CHECKPOINT UTILS (SAFE)
# -------------------------
def _safe_obj_getattribute(obj, name: str):
    try:
        return object.__getattribute__(obj, name)
    except Exception:
        return None


def _infer_parity_seq_len(state_dict):
    for k in ("output_projector.0.weight", "output_projector.weight"):
        if k in state_dict:
            return state_dict[k].shape[0] // 2
    raise RuntimeError("Cannot infer parity_sequence_length")


def _infer_lags(state_dict):
    lags = set()
    for k in state_dict:
        if ".lag_" in k:
            try:
                lags.add(int(k.split(".lag_", 1)[1].split(".", 1)[0]))
            except Exception:
                pass
    return sorted(lags)


def _extract_args_dict(checkpoint, ckpt_path):
    raw = checkpoint.get("args", None)

    if isinstance(raw, dict):
        return dict(raw)

    raw_dict = _safe_obj_getattribute(raw, "__dict__")
    if isinstance(raw_dict, dict) and "_items" in raw_dict:
        return dict(raw_dict["_items"])

    raise RuntimeError(
        f"Checkpoint {ckpt_path} contains unrecoverable args.\n"
        f"Re-save checkpoint with args=dict(wandb.config)."
    )


def load_model_from_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    is_lagged = any("decay_params_out.lag_" in k for k in state)
    args = _extract_args_dict(ckpt, path)

    if "parity_sequence_length" not in args:
        args["parity_sequence_length"] = _infer_parity_seq_len(state)
    if is_lagged and "lags" not in args:
        args["lags"] = _infer_lags(state) or [0, 1, 2, 4]

    required = [
        "iterations", "d_model", "d_input", "heads",
        "n_synch_out", "n_synch_action", "synapse_depth",
        "memory_length", "memory_hidden_dims"
    ]
    for k in required:
        if k not in args:
            raise RuntimeError(f"Missing arg: {k}")

    out_dims = args["parity_sequence_length"] * 2
    reshaper = [args["parity_sequence_length"], 2]

    common = dict(
        iterations=args["iterations"],
        d_model=args["d_model"],
        d_input=args["d_input"],
        heads=args["heads"],
        n_synch_out=args["n_synch_out"],
        n_synch_action=args["n_synch_action"],
        synapse_depth=args["synapse_depth"],
        memory_length=args["memory_length"],
        memory_hidden_dims=args["memory_hidden_dims"],
        backbone_type="parity_backbone",
        positional_embedding_type="custom-rotational-1d",
        out_dims=out_dims,
        prediction_reshaper=reshaper,
        do_layernorm_nlm=False,
    )

    if is_lagged:
        model = LaggedContinuousThoughtMachine(**common, lags=args["lags"])
    else:
        model = ContinuousThoughtMachine(**common)

    model.to(device).eval()

    with torch.no_grad():
        dummy = ParityDataset(args["parity_sequence_length"], 1)[0][0]
        _ = model(dummy.unsqueeze(0).to(device))

    model.load_state_dict(state)
    model.eval()

    return model, args


# -------------------------
# EVAL
# -------------------------
def evaluate(model, seq_len, n_samples, batch, device):
    ds = ParityDataset(seq_len, n_samples)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch)

    acc = []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p, c, _ = model(x)
            p = p.reshape(p.size(0), -1, 2, p.size(-1))
            _, w = parity_loss(p, c, y)
            preds = p.argmax(2)[torch.arange(p.size(0)), :, w]
            acc.append((preds == y).float().mean().item())
    return float(np.mean(acc))


# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()
    set_seed_local(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name="generalization_test",
        config=dict(vars(args)),
        settings=wandb.Settings(start_method="thread", disable_git=True),
    )

    lagged, lagged_args = load_model_from_checkpoint(args.lagged_checkpoint, device)
    base, _ = load_model_from_checkpoint(args.baseline_checkpoint, device)

    results = {}

    for L in args.sequence_lengths:
        set_seed_local(args.seed)
        results[L] = {
            "baseline": evaluate(base, L, args.n_test_samples, args.batch_size, device),
            "lagged": evaluate(lagged, L, args.n_test_samples, args.batch_size, device),
        }
        wandb.log({
            f"baseline/acc_{L}": results[L]["baseline"],
            f"lagged/acc_{L}": results[L]["lagged"],
        })

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()

"""Experiment runner for Optuna-based TuckerRegressorBCD and benchmarks on synthetic data.

Runs hyperparameter optimization with Optuna across independent variable settings
(sample sizes n_train, sparsity cardinalities) and seeds, parallelized across GPUs or CPUs.
"""

import argparse
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from pprint import pprint
import sys
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple

from dotenv import load_dotenv
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import wandb

from benchmark_models import run_benchmark
from hp_study_utils import get_data
from synthetic_study_utils import register_resolvers
from tregs_optuna_searchers import run_tucker_optuna_suite


def parse_and_load_exp_config():
    """Parse CLI arguments and load OmegaConf experiment configuration."""
    description = (
        "Optuna-driven `TuckerRegressorBCD` (v5) experiments on synthetic tensor "
        "regression tasks. Configured via YAML."
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Filename of the YAML configuration file in experiments/synthetic/configs/.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        required=True,
        help='Comma-separated list of seeds, e.g. "0,1,2" or "8".',
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Override experiment name in header.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV results file if it already exists.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help='Comma-separated list of devices, e.g. "cuda:0,cuda:1" or "cpu".',
    )
    parser.add_argument(
        "--no-dask",
        action="store_true",
        help="Run sequentially without starting a Dask distributed cluster.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of Dask workers to spawn. Defaults to number of devices.",
    )

    args, unknown_args = parser.parse_known_args()

    # Determine devices
    if args.devices is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
    else:
        devices = [d.strip() for d in args.devices.split(",")]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Locate config file
    script_path = Path(__file__).parent.resolve()
    cfg_path = script_path / "configs" / args.config
    results_dir = script_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir = results_dir / "optuna_studies"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.is_file():
        print(f"Error: Configuration file {cfg_path} does not exist.")
        sys.exit(1)

    register_resolvers()
    config = OmegaConf.load(cfg_path)

    if args.exp_name is not None:
        config.hdr.name = args.exp_name

    if len(unknown_args) > 0:
        print("Overriding configuration with CLI arguments:")
        for arg in unknown_args:
            print(f"  {arg}")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))

    print("=" * 80)
    print("Optuna Experiment Configuration:")
    print("-" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    return (
        config,
        seeds,
        results_dir,
        optuna_dir,
        devices,
        args.no_dask,
        args.overwrite,
        args.n_workers,
    )


def append_df_to_csv(df: pd.DataFrame, file_path: Path):
    """Safely append a DataFrame to CSV ensuring strict column alignment."""
    if df is None or len(df) == 0:
        return
    if not file_path.exists() or os.stat(file_path).st_size == 0:
        df.to_csv(file_path, mode="w", index=False)
    else:
        existing_header = pd.read_csv(file_path, nrows=0).columns.tolist()
        combined_cols = list(existing_header)
        for col in df.columns:
            if col not in combined_cols:
                combined_cols.append(col)
        if combined_cols != existing_header:
            existing_df = pd.read_csv(file_path)
            merged = pd.concat([existing_df, df], ignore_index=True)
            merged.to_csv(file_path, index=False)
        else:
            aligned_df = df.reindex(columns=existing_header)
            aligned_df.to_csv(file_path, mode="a", index=False, header=False)


def run_single_trial(
    config: Any,
    wandb_kwargs: Optional[Dict[str, Any]],
    trial_control_vars: Dict[str, Any],
    device: str,
    optuna_dir: Path,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Execute all models for a specific seed and IV setting on the target device."""
    register_resolvers()
    data_cfg = config.data

    # Generate synthetic data
    data = get_data(data_cfg)
    n_train = data_cfg.n_train

    X_train = data["X"][:n_train].to(device)
    y_train = data["y"][:n_train].to(device).reshape(-1, 1)

    X_test = data["X_test"].to(device) if data.get("X_test") is not None else None
    y_test = (
        data["y_test"].to(device).reshape(-1, 1)
        if data.get("y_test") is not None
        else None
    )

    B_gt = data.get("B_gt")
    if B_gt is not None:
        B_gt = torch.as_tensor(B_gt).to(device)
    Us_gt = (
        [torch.as_tensor(u).to(device) for u in data["Us_gt"]]
        if data.get("Us_gt") is not None
        else None
    )

    X_val = data.get("X_val")
    y_val = data.get("y_val")
    if X_val is not None:
        X_val = X_val.to(device)
    if y_val is not None:
        y_val = y_val.to(device).reshape(-1, 1)

    dfs: list[pd.DataFrame] = []

    # 1. Run Optuna Tucker suite (TuckerRegs, RidgeTuckerRegs, SparseRTuckerRegs)
    tucker_models = {
        k: OmegaConf.to_object(v)
        for k, v in config.models.items()
        if k in ("TuckerRegs", "RidgeTuckerRegs", "SparseRTuckerRegs")
    }

    if tucker_models:
        lbfgs_cfg = (
            OmegaConf.to_object(config.lbfgs_cfg)
            if hasattr(config, "lbfgs_cfg")
            else None
        )
        rada_cfg = (
            OmegaConf.to_object(config.rada_rgd_cfg)
            if hasattr(config, "rada_rgd_cfg")
            else None
        )

        tucker_df = run_tucker_optuna_suite(
            models_cfg=tucker_models,
            feature_dims=config.data.dims,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            B_gt=B_gt,
            Us_gt=Us_gt,
            control_vars=trial_control_vars,
            wandb_kwargs=wandb_kwargs,
            device=device,
            optuna_storage_dir=optuna_dir,
            lbfgs_config=lbfgs_cfg,
            rada_cfg=rada_cfg,
            seed=trial_control_vars.get("seed", 0),
        )
        dfs.append(tucker_df)

    # 2. Run optional benchmark models (Lasso, ElasticNet, Ridge)
    benchmark_keys = [
        k for k in config.models.keys()
        if k not in ("TuckerRegs", "RidgeTuckerRegs", "SparseRTuckerRegs")
    ]
    for b_key in benchmark_keys:
        m_cfg = config.models[b_key]
        benchmark_cfg = OmegaConf.to_object(m_cfg.get("benchmark_cfg", {}))

        # Train/val split for benchmark
        if X_val is None:
            n_val = max(1, int(n_train * 0.2))
            X_tr, y_tr = X_train[n_val:], y_train[n_val:]
            X_v, y_v = X_train[:n_val], y_train[:n_val]
        else:
            X_tr, y_tr = X_train, y_train
            X_v, y_v = X_val, y_val

        b_df = run_benchmark(
            model_name=m_cfg.model_name,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
            X_test=X_test,
            y_test=y_test,
            B_gt=B_gt,
            control_vars=trial_control_vars,
            benchmark_cfg=benchmark_cfg,
        )
        dfs.append(b_df)

    combined_trial_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined_trial_df, device, trial_control_vars


def main():
    (
        exp_config,
        seeds,
        results_dir,
        optuna_dir,
        devices,
        no_dask,
        overwrite,
        n_workers,
    ) = parse_and_load_exp_config()

    ivs = exp_config.ivs
    iv_names = list(ivs.variables)
    iv_ranges = [list(ivs.ranges[name]) for name in iv_names]

    results_file = results_dir / f"{exp_config.hdr.group}-{exp_config.hdr.name}.csv"
    if overwrite and results_file.exists():
        print(f"Overwriting existing results file: {results_file}")
        results_file.unlink()

    # Build trial jobs
    jobs = []
    for seed in seeds:
        control_vars_base = {"seed": seed}
        for i, vals_tuple in enumerate(product(*iv_ranges)):
            exp_cfg = deepcopy(exp_config)
            exp_cfg.cvs.seed = seed
            exp_cfg.data.seed = seed

            trial_control_vars = deepcopy(control_vars_base)
            for j, iv_name in enumerate(iv_names):
                for cfg_key in exp_cfg.ivs.cfg_keys[iv_name]:
                    OmegaConf.update(exp_cfg, cfg_key, vals_tuple[j])
                trial_control_vars[iv_name] = vals_tuple[j]

            wandb_kwargs = {
                "project": exp_config.hdr.project,
                "group": exp_config.hdr.group,
                "name": f"{exp_config.hdr.name}-s{seed}-n{trial_control_vars.get('n_train')}",
                "config": {**trial_control_vars, **OmegaConf.to_object(exp_cfg.cvs)},
                "tags": list(exp_config.hdr.get("tags", [])) + ["optuna"],
                "reinit": "create_new",
            }

            job = [
                run_single_trial,
                deepcopy(exp_cfg),
                deepcopy(wandb_kwargs),
                deepcopy(trial_control_vars),
                optuna_dir,
            ]
            jobs.append(job)

    print(
        f"\nTotal trials in queue: {len(jobs)} "
        f"across {len(devices)} device(s): {devices}"
    )

    # Execution: Dask or Sequential
    if not no_dask and len(devices) > 1:
        try:
            from dask.distributed import Client, as_completed

            num_workers = (
                n_workers if n_workers is not None else min(len(devices), len(jobs))
            )
            client = Client(n_workers=num_workers, threads_per_worker=1)
            print(f"Dask distributed client initialized with {num_workers} workers.")

            futures = []
            for i in range(min(len(devices), len(jobs))):
                job = jobs.pop(0)
                dev = devices[i % len(devices)]
                func, exp_cfg, w_kwargs, c_vars, o_dir = job
                print(f"[Dispatch] Launching trial on {dev} with {c_vars}...")
                futures.append(client.submit(func, exp_cfg, w_kwargs, c_vars, dev, o_dir))

            ac = as_completed(futures)
            for future in ac:
                df, dev, c_vars = future.result()
                if df is not None and len(df) > 0:
                    append_df_to_csv(df, results_file)
                    print(
                        f"[Completed] Logged {len(df)} model rows from {dev} to "
                        f"{results_file.name} ({c_vars})"
                    )

                if len(jobs) > 0:
                    next_job = jobs.pop(0)
                    func, exp_cfg, w_kwargs, c_vars, o_dir = next_job
                    print(f"[Dispatch] Launching trial on {dev} with {c_vars}...")
                    ac.add(client.submit(func, exp_cfg, w_kwargs, c_vars, dev, o_dir))

            client.close()
            return
        except Exception as e:
            print(f"Dask execution encountered an issue ({e}), falling back to sequential...")

    # Sequential execution
    for job in jobs:
        func, exp_cfg, w_kwargs, c_vars, o_dir = job
        device = devices[0]
        print(f"\n[Dispatch] Running trial on {device} with {c_vars}...")
        df, dev, c_vars = func(exp_cfg, w_kwargs, c_vars, device, o_dir)
        if df is not None and len(df) > 0:
            append_df_to_csv(df, results_file)
            print(f"[Completed] Logged {len(df)} rows to {results_file.name}")

    print(f"\nExperiment completed successfully! Results written to: {results_file}")


if __name__ == "__main__":
    register_resolvers()
    load_dotenv()
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB_INIT_TIMEOUT"] = "600"
        try:
            wandb.login()
        except Exception:
            pass
    main()


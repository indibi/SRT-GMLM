"""Experiment runner for TuckerRegressorBCD and benchmark GLMs on synthetic data.

Runs hyper-parameter search (TuckerBCDRegularizerSearch) and benchmark models
across independent variable settings (sample size, sparsity cardinality) and seeds,
parallelized across GPUs or CPUs.
"""

import os
import sys
import argparse
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence
from time import perf_counter
from copy import deepcopy
from itertools import product
from pprint import pprint
from pathlib import Path

import omegaconf as oc
from omegaconf import OmegaConf
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from synthetic_study_utils import register_resolvers
from hp_study_utils import get_data, get_model_v5
from benchmark_models import run_benchmark

from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd_v5 import TuckerRegressorBCD
from t_regs.models.regression.tucker.treg_nn_bcd_v5_model_selection import TuckerBCDRegularizerSearchV5 as TuckerBCDRegularizerSearch


def parse_and_load_exp_config():
    description = (
        "`TuckerRegressorBCD` (v5) with Regularizer Search and Benchmark GLM experiments "
        "on synthetic tensor regression tasks. Configured via YAML."
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Filename of the YAML configuration file in experiments/synthetic/configs/.'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        required=True,
        help='Comma-separated list of seeds, e.g. "0,1,2" or "8".'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='Override experiment name in header.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing CSV results file if it already exists.'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default=None,
        help='Comma-separated list of devices, e.g. "cuda:0,cuda:1" or "cpu".'
    )
    parser.add_argument(
        '--no-dask',
        action='store_true',
        help='Run sequentially without starting a Dask distributed cluster.'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Number of Dask workers to spawn. Defaults to number of devices.'
    )

    args, unknown_args = parser.parse_known_args()

    # Determine devices
    if args.devices is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            devices = ['cpu']
    else:
        devices = [d.strip() for d in args.devices.split(',')]

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Locate config file
    script_path = Path(__file__).parent.resolve()
    cfg_path = script_path / 'configs' / args.config
    results_dir = script_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

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
    print("Experiment Configuration:")
    print("-" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    return config, seeds, results_dir, devices, args.no_dask, args.overwrite, args.n_workers


def run_exp_for_ivs_and_model(config, wandb_kwargs, control_vars, model_key, device):
    """Execute a single experimental trial on a specified device."""
    register_resolvers()
    model_cfg = config.models[model_key]
    data_cfg = config.data

    # Generate synthetic dataset
    data = get_data(data_cfg)
    n_train = data_cfg.n_train

    X_train = data['X'][:n_train].to(device)
    y_train = data['y'][:n_train].to(device).reshape(-1, 1)

    X_test = data['X_test'].to(device) if data.get('X_test') is not None else None
    y_test = data['y_test'].to(device).reshape(-1, 1) if data.get('y_test') is not None else None

    B_gt = data.get('B_gt')
    if B_gt is not None:
        B_gt = torch.as_tensor(B_gt).to(device)
    Us_gt = [torch.as_tensor(u).to(device) for u in data['Us_gt']] if data.get('Us_gt') is not None else None

    # Optional separate validation set if specified in data
    X_val = data.get('X_val')
    y_val = data.get('y_val')
    if X_val is not None:
        X_val = X_val.to(device)
    if y_val is not None:
        y_val = y_val.to(device).reshape(-1, 1)

    # 1. Main TuckerBCDRegularizerSearch (v5)
    if model_cfg.model_name in ('TuckerBCDRegularizerSearch', 'tucker_bcd_search'):
        regsearch_args = OmegaConf.to_object(model_cfg.regsearch_args)
        regsearch = TuckerBCDRegularizerSearch(**regsearch_args)

        base_model_cfg = deepcopy(model_cfg.base_model_cfg)
        base_model_cfg.regressor.feature_ranks = model_cfg.feature_ranks

        base_model = get_model_v5(base_model_cfg, device=device)
        regsearch.search(
            base_model=base_model,
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            B_gt=B_gt,
            Us_gt=Us_gt,
            control_vars=control_vars,
            wandb_kwargs=wandb_kwargs,
            seed=control_vars.get('seed', 0),
        )
        df: pd.DataFrame = regsearch.summary_df

    # 2. Benchmark Models
    else:
        benchmark_cfg = OmegaConf.to_object(model_cfg.get('benchmark_cfg', {}))
        # If no explicit X_val, create a default split for benchmark parameter selection
        if X_val is None:
            n_val = max(1, int(n_train * 0.2))
            X_tr = X_train[n_val:]
            y_tr = y_train[n_val:]
            X_v = X_train[:n_val]
            y_v = y_train[:n_val]
        else:
            X_tr, y_tr, X_v, y_v = X_train, y_train, X_val, y_val

        df = run_benchmark(
            model_name=model_cfg.model_name,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
            X_test=X_test,
            y_test=y_test,
            B_gt=B_gt,
            control_vars=control_vars,
            benchmark_cfg=benchmark_cfg,
        )

    return df, device, model_key, control_vars


def append_df_to_csv(df: pd.DataFrame, file_path: Path):
    """Safely append a DataFrame to CSV ensuring strict column alignment."""
    if df is None or len(df) == 0:
        return
    if not file_path.exists() or os.stat(file_path).st_size == 0:
        df.to_csv(file_path, mode='w', index=False)
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
            aligned_df.to_csv(file_path, mode='a', index=False, header=False)


def main():
    exp_config, seeds, results_dir, devices, no_dask, overwrite, n_workers = parse_and_load_exp_config()
    ivs = exp_config.ivs
    iv_names = list(ivs.variables)
    iv_ranges = [list(ivs.ranges[name]) for name in iv_names]

    results_file = results_dir / f"{exp_config.hdr.group}-{exp_config.hdr.name}.csv"
    if overwrite and results_file.exists():
        print(f"Overwriting existing results file: {results_file}")
        results_file.unlink()

    # Build prioritized job queues:
    # Heavy GPU-bound Tucker models are queued first to saturate all GPUs simultaneously.
    # Fast CPU benchmark models (Lasso/ElasticNet/Ridge) are queued second.
    tucker_jobs = []
    benchmark_jobs = []

    for seed in seeds:
        control_vars_base = {'seed': seed}
        for i, vals_tuple in enumerate(product(*iv_ranges)):
            exp_cfg = deepcopy(exp_config)
            exp_cfg.cvs.seed = seed
            exp_cfg.data.seed = seed

            trial_control_vars = deepcopy(control_vars_base)
            for j, iv_name in enumerate(iv_names):
                for cfg_key in exp_cfg.ivs.cfg_keys[iv_name]:
                    OmegaConf.update(exp_cfg, cfg_key, vals_tuple[j])
                trial_control_vars[iv_name] = vals_tuple[j]

            for model_key, m_cfg in exp_cfg.models.items():
                wandb_kwargs = {
                    'project': exp_config.hdr.project,
                    'group': exp_config.hdr.group,
                    'name': f"{exp_config.hdr.name}-s{seed}-iv{i}-{model_key}",
                    'config': {**trial_control_vars, **OmegaConf.to_object(exp_cfg.cvs)},
                    'reinit': 'create_new',
                }

                job = [
                    run_exp_for_ivs_and_model,
                    deepcopy(exp_cfg),
                    deepcopy(wandb_kwargs),
                    deepcopy(trial_control_vars),
                    model_key,
                ]

                if m_cfg.model_name in ('TuckerBCDRegularizerSearch', 'tucker_bcd_search'):
                    tucker_jobs.append(job)
                else:
                    benchmark_jobs.append(job)

    # Heavy Tucker jobs first, then fast benchmarks
    jobs_in_queue = tucker_jobs + benchmark_jobs

    print(
        f"\nTotal tasks in queue: {len(jobs_in_queue)} "
        f"({len(tucker_jobs)} Tucker tensor jobs, {len(benchmark_jobs)} benchmark jobs) "
        f"across {len(devices)} devices: {devices}"
    )

    # Execution: Dask or Sequential
    if not no_dask and len(devices) > 1:
        try:
            from dask.distributed import Client, as_completed
            num_workers = n_workers if n_workers is not None else min(len(devices), len(jobs_in_queue))
            client = Client(n_workers=num_workers, threads_per_worker=1)
            print(f"Dask distributed client initialized with {num_workers} workers.")

            futures = []
            for i in range(min(len(devices), len(jobs_in_queue))):
                job = jobs_in_queue.pop(0)
                dev = devices[i % len(devices)]
                print(f"[Dispatch] Launching '{job[4]}' on {dev} with {job[3]}...")
                futures.append(client.submit(*job, dev))

            ac = as_completed(futures)
            for future in ac:
                res = future.result()
                if isinstance(res, tuple) and len(res) == 4:
                    df, dev, m_key, c_vars = res
                else:
                    df, dev = res[0], res[1]
                    m_key, c_vars = "model", {}

                if df is not None:
                    append_df_to_csv(df, results_file)
                    print(f"[Completed] [{m_key}] Logged {len(df)} rows from {dev} to {results_file.name} ({c_vars})")

                if len(jobs_in_queue) > 0:
                    next_job = jobs_in_queue.pop(0)
                    print(f"[Dispatch] Launching '{next_job[4]}' on {dev} with {next_job[3]}...")
                    ac.add(client.submit(*next_job, dev))

            client.close()
            return
        except Exception as e:
            print(f"Dask execution encountered an issue ({e}), falling back to direct execution...")

    # Sequential / In-process execution fallback
    for job in jobs_in_queue:
        func, exp_cfg, wandb_kwargs, trial_control_vars, model_key = job
        device = devices[0]
        print(f"\n[Dispatch] Running {model_key} on {device} with {trial_control_vars}...")
        res = func(exp_cfg, wandb_kwargs, trial_control_vars, model_key, device)
        if isinstance(res, tuple) and len(res) == 4:
            df, dev, m_key, c_vars = res
        else:
            df, dev = res[0], res[1]
            m_key, c_vars = model_key, trial_control_vars

        if df is not None:
            append_df_to_csv(df, results_file)
            print(f"[Completed] [{m_key}] Logged {len(df)} rows to {results_file.name}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    register_resolvers()
    load_dotenv()
    if 'WANDB_API_KEY' in os.environ:
        os.environ['WANDB_INIT_TIMEOUT'] = '600'
        try:
            wandb.login()
        except Exception:
            pass
    main()

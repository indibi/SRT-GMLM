from collections import defaultdict
from typing import Any, Dict, Optional, Callable
from time import perf_counter
from copy import deepcopy
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
from pathlib import Path

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
import omegaconf as oc
from omegaconf import OmegaConf

from t_regs.manifolds import Steifel, Euclidean, GeneralizedSteifel
from t_regs.solvers.manifold.minimax_problem import MinimaxProblem
from t_regs.solvers.manifold.rada import RADA_RGD
from t_regs.solvers.manifold.line_searcher import ArmijoPointSearch
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd_v2 import TuckerRegressorBCD
from t_regs.synthetic_data.generate_lr_data import (
    generate_sparse_low_tucker_rank_tensor,
    generate_low_rank_data,
    generate_smooth_low_tucker_rank_tensor
    )
from t_regs.synthetic_data.regression import (
    generate_tensor_regression_predictors,
    generate_tensor_regression_response
    )


def get_data(data_cfg: oc.DictConfig) -> dict[str, Any]:
    """Generate synthetic data for regression problems according to the configuration."""
    tensor_type = data_cfg.tensor_type
    n_samples = data_cfg.n_train + data_cfg.n_val + data_cfg.get('n_test', 0)
    dims = data_cfg.dims
    seed = data_cfg.seed
    if tensor_type == 'low_tucker_rank':
        B_gt = generate_low_rank_data(
            dims=dims,
            rank=data_cfg.ranks,
            seed=seed
            )
    elif tensor_type == 'sparse_low_tucker_rank':
        B_gt = generate_sparse_low_tucker_rank_tensor(
            dims=dims,
            ranks=data_cfg.ranks,
            cardinalities=data_cfg.cardinalities,
            seed=seed
        )
    elif tensor_type == 'smooth_low_tucker_rank':
        raise NotImplementedError(
            "Smooth low Tucker rank tensor generation is not implemented yet."
            )
    else:
        raise ValueError(f"Unknown tensor type {tensor_type}.")
    X = generate_tensor_regression_predictors(
        n_samples=n_samples,
        pred_dims=dims,
        seed=seed
    )
    y = generate_tensor_regression_response(
        X=X,
        B=B_gt,
        noise_type=data_cfg.get('noise_type', 'gaussian'),
        regression_type=data_cfg.regression_type,
        seed=seed,
        kernel=data_cfg.get('kernel', None),
        noise_std=data_cfg.get('noise_std', 0.1)
    )
    # Split into train/val/test
    n_train = data_cfg.n_train
    n_val = data_cfg.n_val
    X_train = torch.tensor(X[:n_train], dtype=torch.double)
    X_val = torch.tensor(X[n_train:n_train+n_val], dtype=torch.double)
    y_train = torch.tensor(y[:n_train], dtype=torch.double)
    y_val = torch.tensor(y[n_train:n_train+n_val], dtype=torch.double)
    return {
        'B_gt': B_gt,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
    }


def build_model_and_solver(model_cfg: oc.DictConfig, device: str) -> TuckerRegressorBCD:
    """Build TuckerRegressorBCD from config."""
    dims = model_cfg.dims
    ranks = model_cfg.ranks
    ldas = model_cfg.get('ldas', [0.0] * len(dims))
    lda_core = model_cfg.get('lda_core', 0.0)
    tau = model_cfg.get('tau', 0.0001)
    regression_type = model_cfg.regression_type

    core_dim = math.prod(ranks)

    # Create manifolds
    feature_manifolds = [Steifel(n, p, device=device) for n, p in zip(dims, ranks)]
    model = TuckerRegressor(
        regression_type,
        dims,
        ranks,
        task_dims=tuple([]),
        feature_manifolds=feature_manifolds
    ).to(device)

    # Create subproblem solvers
    subproblem_solvers = {}
    for mode in range(1, len(dims) + 1):
        lda = ldas[mode-1]
        fd = dims[mode-1]
        fr = ranks[mode-1]
        subproblem_solvers[f'U_{mode}'] = RADA_RGD(
            R=lda * (fd * fr)**0.5 if lda != 0 else 1e-6 * (fd * fr)**0.5,
            c1=1e-1,
            max_it=400,
            Tk=5,
            max_line_search=10,
            beta1=0.01 * fd * fr**0.5,
            rho=2.0,
            tau_1=0.999,
            tau_2=0.9,
            zeta=1.0,
            zeta_BB_min=1e-20,
            zeta_BB_max=1e20,
            eps=1e-8,
            max_time=None,
            max_function_evals=10 * 1000 * 5,
            verbosity=0,
        )
    subproblem_solvers['C'] = RADA_RGD(
        R=lda_core * (core_dim)**0.5 if lda_core != 0 else 1e-6 * (core_dim)**0.5,
        c1=1e-1,
        max_it=400,
        Tk=5,
        max_line_search=25,
        beta1=0.01 * (core_dim)**0.5,
        rho=2.0,
        tau_1=0.999,
        tau_2=0.9,
        zeta=1.0,
        zeta_BB_min=1e-20,
        zeta_BB_max=1e20,
        eps=1e-8,
        max_time=None,
        max_function_evals=10 * 1000 * 5,
        verbosity=0,
    )

    Ds = [torch.eye(d, device=device, dtype=torch.double) for d in dims]

    solver = TuckerRegressorBCD(
        model,
        tau=tau,
        ldas=tuple(ldas),
        lda_core=lda_core,
        thetas=tuple([0.0] * len(dims)),
        Ds=Ds,
        Ls=[None] * len(dims),
        subproblem_solvers=subproblem_solvers,
        main_algorithm='BCD_RADA_RGD',
        verbosity=0,
        device=device,
        init_with_hosvd_of_grad=True,
        max_it=50,
        max_time=380913820913820,
        min_gradient_norm=1e-4,
    )
    
    return solver


def evaluate_rank_candidate(
    ranks: tuple,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model_cfg: oc.DictConfig,
    device: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate a rank candidate using cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_scores = []
    val_scores = []
    
    cfg = deepcopy(model_cfg)
    cfg['ranks'] = ranks
    
    for fold_idx, (train_idx, fold_val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train[train_idx].to(device)
        y_fold_train = y_train[train_idx].to(device)
        X_fold_val = X_train[fold_val_idx].to(device)
        y_fold_val = y_train[fold_val_idx].to(device)
        
        solver = build_model_and_solver(cfg, device)
        solver.fit(X_fold_train, y_fold_train, X_val=X_fold_val, y_val=y_fold_val)
        
        train_scores.append(solver.log['iterations']['train_score'][-1])
        val_scores.append(solver.log['iterations']['val_score'][-1])
    
    return {
        'ranks': ranks,
        'device': device,
        'mean_train_score': np.mean(train_scores),
        'mean_val_score': np.mean(val_scores),
        'std_train_score': np.std(train_scores),
        'std_val_score': np.std(val_scores),
        'train_scores': train_scores,
        'val_scores': val_scores,
    }


def parallel_rank_search(
    model_cfg: oc.DictConfig,
    data: dict,
    rank_candidates: list,
    devices: list,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perform parallel hyperparameter search over rank candidates."""
    
    def evaluate_with_device(ranks, device):
        return evaluate_rank_candidate(
            ranks,
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            model_cfg,
            device,
            n_splits=n_splits,
            random_state=random_state,
        )
    
    results = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for idx, ranks in enumerate(rank_candidates):
            device = devices[idx % len(devices)]
            future = executor.submit(evaluate_with_device, ranks, device)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed: ranks={result['ranks']}, "
                  f"val_score={result['mean_val_score']:.4f} ± {result['std_val_score']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_val_score', ascending=False).reset_index(drop=True)
    return results_df


def main():
    """Main entry point for parallel rank search."""
    parser = argparse.ArgumentParser(
        description='Parallel hyperparameter rank search for Tucker Regressor'
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to YAML configuration file (relative to configs/ directory)'
    )
    parser.add_argument(
        '--rank-candidates',
        type=str,
        default='(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6)',
        help='Comma-separated rank candidates (default: small ranks for testing)'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='cuda:0,cuda:1,cuda:2,cuda:3',
        help='Comma-separated CUDA devices (default: cuda:0-3)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of K-fold splits (default: 5)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / 'configs' / args.config_file
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    print(OmegaConf.to_yaml(cfg))
    
    # Parse rank candidates
    rank_candidates = []
    for rank_str in args.rank_candidates.split(','):
        rank_str = rank_str.strip()
        if rank_str.startswith('(') and rank_str.endswith(')'):
            rank_str = rank_str[1:-1]
        ranks = tuple(int(r.strip()) for r in rank_str.split(','))
        rank_candidates.append(ranks)
    
    # Parse devices
    devices = [d.strip() for d in args.devices.split(',')]
    
    print(f"\nRank candidates: {rank_candidates}")
    print(f"Devices: {devices}")
    print(f"K-fold splits: {args.n_splits}\n")
    
    # Generate data
    print("Generating synthetic data...")
    data = get_data(cfg.data)
    print(f"Data shapes: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}")
    print(f"              X_val={data['X_val'].shape}, y_val={data['y_val'].shape}\n")
    
    # Perform parallel search
    print("Starting parallel rank search...")
    tic = perf_counter()
    results_df = parallel_rank_search(
        cfg.models.Tucker,
        data,
        rank_candidates,
        devices,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    elapsed = perf_counter() - tic
    
    print(f"\nSearch completed in {elapsed:.2f} seconds")
    print("\nResults (sorted by validation score):")
    print(results_df[['ranks', 'device', 'mean_train_score', 'mean_val_score', 'std_val_score']])
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_name = Path(args.config_file).stem
    results_file = output_dir / f"rank_search_{config_name}_{int(tic)}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()


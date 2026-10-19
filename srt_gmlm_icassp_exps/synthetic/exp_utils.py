# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, Optional, Callable
from time import perf_counter


import wandb
from dask.distributed import Client, as_completed
import networkx as nx
import torch
import numpy as np
# import numpy.linalg as npla
# import pandas as pd
# from sklearn.metrics import (f1_score, precision_score, recall_score, 
#                              roc_auc_score, precision_recall_curve, auc,
#                              roc_curve)
from sklearn.model_selection import KFold
# from tqdm import tqdm
import optuna
import omegaconf as oc
from omegaconf import OmegaConf

# from t_regs.manifolds import Steifel, Euclidean, GeneralizedSteifel
# from t_regs.solvers.manifold.minimax_problem import MinimaxProblem
# from t_regs.solvers.manifold.rada import RADA_RGD
# from t_regs.solvers.manifold import RiemmannianGradientDescent as RGD
# from t_regs.solvers.manifold.line_searcher import ArmijoPointSearch
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

class Experiment:

    def __init__(
        self,
        config: oc.DictConfig,
        devices: list[str] = ['cuda' if torch.cuda.is_available() else 'cpu'],
        overwrite: bool = False,
        ):
        self.config = config
        self.devices = devices

        wandb.login()

    def run_experiment(self):
        project_name = self.config.experiment_header.project_name
        group_name = self.config.experiment_header.experiment_name # group_name
        tags = self.config.experiment_header.tags
        notes = self.config.experiment_header.notes

        models = self.config.experiment_config.models
        m_keys = models.keys()

        ind_vars = self.config.experiment_config.independent_variables
        for m_key in m_keys:
            m_cfg = models[m_key]

            with wandb.init(
                project=project_name,
                group=group_name,
                tags=tags,
                name=
                ) as run:







def run_model_data(exp_config, model_key, ):

    hp_study = model_cfg.hp_study_cfg
    if hp_study is None:

    else:
        pass


def fit_and_evaluate(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_cfg: oc.DictConfig,
        device: str,
        ):
    OmegaConf.update(model_cfg, 'device', device)
    model = get_model(model_cfg, device)
    X_train = X_train.to(device=device) # torch.tensor(X_train, dtype=torch.double, device=device)
    y_train = y_train.to(device=device) # torch.tensor(y_train, dtype=torch.double, device=device)
    X_val = X_val.to(device=device) # torch.tensor(X_val, dtype=torch.double, device=device)
    y_val = y_val.to(device=device) # torch.tensor(y_val, dtype=torch.double, device=device)

    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val
        )

    return {
        'train_score': model.log['iterations']['train_score'][-1],
        'val_score': model.log['iterations']['val_score'][-1]
    }

def get_model(
    model_cfg: oc.DictConfig,
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> TuckerRegressorBCD:
    model_cfg = model_cfg.copy()
    model_cfg.device = device
    model_cfg = OmegaConf.to_object(model_cfg)
    # regressor_cfg = OmegaConf.to_object(model_cfg.regressor)
    regressor_cfg = model_cfg['regressor']
    regressor = TuckerRegressor(**regressor_cfg)

    return TuckerRegressorBCD(
        regressor,
        tau = model_cfg['tau'],
        ldas = model_cfg['ldas'],
        lda_core = model_cfg['lda_core'],
        thetas = model_cfg['thetas'],
        Ds = model_cfg['Ds'],
        Ls = model_cfg['Ls'],
        subproblem_solvers = model_cfg['subproblem_solvers'],
        dataloader_cfg = model_cfg['dataloader_cfg'],
        main_algorithm = model_cfg['main_algorithm'],
        verbosity = model_cfg['verbosity'],
        init_with_hosvd_of_grad = model_cfg['init_with_hosvd_of_grad'],
        max_it = model_cfg['max_it'],
        max_time = model_cfg['max_time'],
        min_gradient_norm = model_cfg['min_gradient_norm'],
        device = device,
    )


def get_data(data_cfg: oc.DictConfig) -> dict[str, np.ndarray]:
    tensor_type = data_cfg.tensor_type
    n_samples = data_cfg.n_train
    dims = data_cfg.dims
    seed = data_cfg.seed

    if tensor_type == 'low_tucker_rank':
        B_gt = generate_low_rank_data(
            dim=dims,
            ranks=data_cfg.ranks,
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
        dims = dims,
        n_samples=n_samples,
        seed=seed,
    )
    y,_ = generate_tensor_regression_response(
        X=X,
        B=B_gt,
        noise_type=None,
        regression_type=data_cfg.regression_type,
        seed=seed,
    )
    X = torch.tensor(X, dtype=torch.double, device='cpu')
    y = torch.tensor(y, dtype=torch.double, device='cpu')
    B_gt = torch.tensor(B_gt, dtype=torch.double, device='cpu')
    return {
        'B_gt': B_gt,
        'X': X,
        'y': y
    }
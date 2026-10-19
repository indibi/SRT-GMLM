# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, Optional, Callable
from time import perf_counter


# import wandb
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
# from t_regs.models.regression.tucker_nn_bcd_v2 import TuckerRegressorBCD
# from t_regs.models.regression.tucker_nn_bcd_v5 import TuckerRegressorBCD as TuckerRegressorBCDV5
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD as TuckerRegressorBCDV5

from t_regs.synthetic_data.generate_lr_data import (
    generate_sparse_low_tucker_rank_tensor,
    generate_low_rank_data,
    generate_smooth_low_tucker_rank_tensor
    )
from t_regs.synthetic_data.regression import (
    generate_tensor_regression_predictors,
    generate_tensor_regression_response
    )


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

def get_model_v5(
    model_cfg: oc.DictConfig,
    device='cuda' if torch.cuda.is_available() else 'cpu'
) -> TuckerRegressorBCDV5:
    model_cfg = model_cfg.copy()
    model_cfg.device = device
    model_dict = OmegaConf.to_object(model_cfg)
    regressor_cfg = model_dict['regressor']
    regressor = TuckerRegressor(**regressor_cfg)

    return TuckerRegressorBCDV5(
        tucker_regressor=regressor,
        tau=model_dict.get('tau', 0.0),
        ldas=model_dict.get('ldas', [0.0] * regressor.covariant_degree),
        thetas=model_dict.get('thetas', [0.0] * regressor.covariant_degree),
        Ds=model_dict.get('Ds', None),
        Ls=model_dict.get('Ls', None),
        min_gradient_norm=model_dict.get('min_gradient_norm', 1e-5),
        max_it=model_dict.get('max_it', 100),
        max_time=model_dict.get('max_time', None),
        subproblem_solvers=model_dict.get('subproblem_solvers', None),
        lbfgs_config=model_dict.get('lbfgs_config', {
            'max_iter': 500,
            'tolerance_grad': 1e-8,
            'tolerance_change': 1e-9,
            'line_search_fn': 'strong_wolfe',
        }),
        verbosity=model_dict.get('verbosity', 0),
        log_verbosity=model_dict.get('log_verbosity', 1),
        dataloader_cfg=model_dict.get('dataloader_cfg', None),
        init_with_hosvd_of_grad=model_dict.get('init_with_hosvd_of_grad', False),
        init_with_lbfgs_full_glm=model_dict.get('init_with_lbfgs_full_glm', True),
        transform_covariates=model_dict.get('transform_covariates', True),
        device=device,
    )

def get_data(data_cfg: oc.DictConfig) -> dict[str, np.ndarray]:
    tensor_type = data_cfg.tensor_type
    n_test = OmegaConf.select(data_cfg, 'n_test', default=0)
    n_val = OmegaConf.select(data_cfg, 'n_val', default=0)
    n_samples = OmegaConf.select(data_cfg, 'n_samples',
        default=data_cfg.n_train + n_test + n_val
        )
    dims = data_cfg.dims
    seed = data_cfg.seed

    if tensor_type == 'low_tucker_rank':
        B_gt, C_gt, Us_gt = generate_low_rank_data(
            dim=dims,
            ranks=data_cfg.ranks,
            seed=seed,
            return_factors=True,
            )
    elif tensor_type == 'sparse_low_tucker_rank':
        B_gt, C_gt, Us_gt = generate_sparse_low_tucker_rank_tensor(
            dims=dims,
            ranks=data_cfg.ranks,
            cardinalities=data_cfg.cardinalities,
            expanded=data_cfg.expanded,
            seed=seed,
            return_factors=True,
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
    n_train = n_samples - (n_val + n_test)
    X_val = None
    y_val = None
    X_test = None
    y_test = None
    if n_val + n_test > 0:
        if n_val > 0:
            X_val = X[n_train:n_train+n_val]
            y_val = y[n_train:n_train+n_val]
        if n_test > 0:
            X_test = X[n_train+n_val:]
            y_test = y[n_train+n_val:]
        X = X[:n_train]
        y = y[:n_train]
    return {
        'B_gt': B_gt,
        'C_gt': C_gt,
        'Us_gt': Us_gt,
        'X': X,
        'y': y,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }

class HyperparameterStudy:
    def __init__(
        self,
        config: oc.DictConfig,
        devices: list[str] = ['cuda' if torch.cuda.is_available() else 'cpu'],
        overwrite: bool = False
        ):
        self.config = config
        self.devices = devices

        study_name = config.study_header.study_name
        storage = config.study_header.storage
        # check if storage exists
        studies = optuna.study.get_all_study_names(storage=storage)
        if study_name in studies and overwrite:
            print(f"Study {study_name} already exists. Deleting it and creating a new one.")
            optuna.delete_study(study_name=study_name, storage=storage)
        if len(config.study_config.directions) > 1:
            self._study = optuna.study.create_study(
                study_name=study_name,
                storage=storage,
                directions=config.study_config.directions,
                sampler=config.study_config.sampler,
                pruner=config.study_config.pruner,
                load_if_exists=True
            )
        else:
            self._study = optuna.study.create_study(
                study_name=study_name,
                storage=storage,
                direction=config.study_config.directions[0],
                sampler=config.study_config.sampler,
                pruner=config.study_config.pruner,
                load_if_exists=True
            )
        objective_names = OmegaConf.to_object(
            config.study_config.objective_config.objective_names
            )
        self._study.set_metric_names(objective_names)
        self._study.set_user_attr(
            'config',
            OmegaConf.to_container(config, resolve=False)
            # OmegaConf.to_yaml(config)
            )

    def run_study(self):
        objective = StudyObjective(
            objective_cfg=self.config.study_config.objective_config,
            model_cfg=self.config.model_cfg,
            data_cfg=self.config.data,
            devices=self.devices
        )
        print(f"Starting hyper-parameter study: {self._study.study_name}.")
        self._study.optimize(
            objective,
            n_trials=self.config.study_config.n_trials,
            show_progress_bar=True,
            )

        best_trial = self._study.best_trial
        print(f"Best trial: {best_trial.number}")
        print(f"Best value: {best_trial.value}")
        print("Best hyper-parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        df = self._study.trials_dataframe()
        df.columns = [col.replace('user_attrs_', '') for col in df.columns]
        results_dir = self.config.study_header.results_dir
        study_name = self.config.study_header.study_name
        group_name = self.config.study_header.group_name
        results_path = f"{results_dir}/{group_name}/{study_name}_results.csv"
        df.to_csv(results_path, index=False)
        print(f"Study results saved to {results_path}.")


class StudyObjective:
    def __init__(self,
        objective_cfg: oc.DictConfig,
        model_cfg: oc.DictConfig,
        data_cfg: oc.DictConfig,
        devices: list[str]
        ):
        self.objective_cfg = objective_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.devices = devices

        self._data = get_data(data_cfg)
        self._n_samples = data_cfg.n_train
        self._indices = np.arange(self._n_samples)
        self._client = Client(n_workers=len(devices), threads_per_worker=2)

        if objective_cfg.split.type == 'k-fold':
            self._splitter = KFold(
                n_splits=objective_cfg.split.n_splits,
                shuffle=False,
            )
        else:
            raise NotImplementedError(
                f"Data splitting method {objective_cfg.split} is not implemented."
            )

    def __call__(self, trial: optuna.Trial) -> float:
        hparams = self._ask_hyper_parameters(trial)
        model_cfg_copy = deepcopy(self.model_cfg)
        
        for hp_key, hp_value in hparams.items():
            if hp_key in self.objective_cfg.hp_to_model_cfg_keys.keys():
                for model_cfg_key in self.objective_cfg.hp_to_model_cfg_keys[hp_key]:
                    OmegaConf.update(model_cfg_copy, model_cfg_key, hp_value)

        if self.objective_cfg.split.type == 'k-fold':
            kf = self._splitter
            indices = self._indices

            results = defaultdict(list)
            # with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            # futures = []
            for f_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
                X_train = self._data['X'][train_idx]
                X_val = self._data['X'][val_idx]
                y_train = self._data['y'][train_idx].reshape(-1,1)
                y_val = self._data['y'][val_idx].reshape(-1,1)
                device = self.devices[f_idx % len(self.devices)]
                # future = self._client.submit(# executor.submit(
                result = fit_and_evaluate(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    model_cfg_copy,
                    device,
                    )
                # futures.append(future)
                for k,v in result.items():
                    results[k].append(v)
                if f_idx == 1:
                    break
            # for future in as_completed(futures):
            #     result = future.result()
            #     for k,v in result.items():
            #         results[k].append(v)
            mean_results = {
                f'mean_{k}': np.mean(results[k]) for k in results.keys()
            }
            std_results = {
                f'std_{k}': np.std(results[k]) for k in results.keys()
            }
            summary_results = {**mean_results, **std_results}
            for k, v in summary_results.items():
                trial.set_user_attr(k, v)
            return summary_results['mean_val_score']
        else:
            raise NotImplementedError(
                f"Data splitting method {self.objective_cfg.split} is not implemented."
            )


    def _ask_hyper_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        hparams = {}
        for hp_key, hp_cfg in self.objective_cfg.distributions.items():
            if hp_cfg['type'] == 'int':
                hparams[hp_key] = trial.suggest_int(
                    hp_key, **hp_cfg['kwargs']
                    )
            elif hp_cfg['type'] == 'float':
                hparams[hp_key] = trial.suggest_float(
                    hp_key, **hp_cfg['kwargs']
                    )
            elif hp_cfg['type'] == 'categorical':
                hparams[hp_key] = trial.suggest_categorical(
                    hp_key, **hp_cfg['kwargs']
                    )
        return hparams



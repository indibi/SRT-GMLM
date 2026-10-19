"""Optuna-based Hyper-parameter Searchers for TuckerRegressorBCD (v5).

Provides modular searcher classes:
  - `TuckerRegsSearcher`: Searches for optimal factor ranks with tau=0, ldas=0.
  - `RidgeTuckerRegsSearcher`: Searches for optimal factor ranks and core ridge penalty tau.
  - `SparseRTuckerRegsSearcher`: Uses best factor ranks from RidgeTuckerRegs and searches
    for core ridge penalty tau and factor sparsity penalties ldas.
  - `run_tucker_optuna_suite`: Coordinates running the 3 models sequentially on a dataset,
    logging to WandB and local Optuna SQLite storage.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import torch
import wandb

from t_regs.manifolds.steifel import Steifel
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd_v5 import TuckerRegressorBCD
from t_regs.solvers.manifold.rada_v3 import RADA_RGD


def compute_classification_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_prob: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Compute binary classification metrics: AU-ROC, AU-PRC, Accuracy, Balanced Acc, F1, Precision, Recall, and Loss."""
    yt = (
        y_true.detach().cpu().numpy().ravel().astype(int)
        if isinstance(y_true, torch.Tensor)
        else np.asarray(y_true).ravel().astype(int)
    )
    yp = (
        y_prob.detach().cpu().numpy().ravel().astype(float)
        if isinstance(y_prob, torch.Tensor)
        else np.asarray(y_prob).ravel().astype(float)
    )

    yp = np.nan_to_num(yp, nan=0.5, posinf=1.0, neginf=0.0)
    yp_clipped = np.clip(yp, 1e-7, 1.0 - 1e-7)
    bce_loss = float(-np.mean(yt * np.log(yp_clipped) + (1.0 - yt) * np.log(1.0 - yp_clipped)))

    y_pred = (yp >= 0.5).astype(int)
    acc = float(accuracy_score(yt, y_pred))
    bal_acc = float(balanced_accuracy_score(yt, y_pred))
    f1 = float(f1_score(yt, y_pred, average="binary", zero_division=0))
    prec = float(precision_score(yt, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(yt, y_pred, average="binary", zero_division=0))

    unique_classes = np.unique(yt)
    if len(unique_classes) > 1:
        try:
            au_roc = float(roc_auc_score(yt, yp))
        except Exception:
            au_roc = float("nan")
        try:
            au_prc = float(average_precision_score(yt, yp))
        except Exception:
            au_prc = float("nan")
    else:
        au_roc = float("nan")
        au_prc = float("nan")

    return {
        "au_roc": au_roc,
        "au_prc": au_prc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "loss": bce_loss,
    }


@torch.no_grad()
def evaluate_gt_recovery(
    model: TuckerRegressorBCD,
    B_gt: Optional[torch.Tensor] = None,
    Us_gt: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute sparsity nnz ratios and ground truth tensor / subspace recovery."""
    metrics: Dict[str, float] = {}
    N = model.tucker_regressor.covariant_degree
    for m in range(1, N + 1):
        u = model.tucker_regressor.Us[m - 1]
        nnz = float((u.abs() >= 1e-5).sum() / u.numel())
        metrics[f"U_{m}_nnz_ratio"] = nnz

    b_hat = model.tucker_regressor.expanded_form()
    metrics["B_nnz_ratio"] = float((b_hat.abs() >= 1e-5).sum() / b_hat.numel())

    if B_gt is not None:
        b_gt_dev = torch.as_tensor(B_gt, device=b_hat.device, dtype=b_hat.dtype)
        err = torch.linalg.norm(b_hat - b_gt_dev) / torch.linalg.norm(b_gt_dev)
        metrics["rel_coef_error"] = float(err.item())

    if Us_gt is not None:
        for m in range(1, N + 1):
            u_hat = model.tucker_regressor.Us[m - 1]
            u_true = torch.as_tensor(Us_gt[m - 1], device=u_hat.device, dtype=u_hat.dtype)
            p_hat = u_hat @ u_hat.T
            p_true = u_true @ u_true.T
            r = u_hat.shape[1]
            dist = torch.linalg.norm(p_hat - p_true) / ((2.0 * r) ** 0.5)
            metrics[f"U_{m}_subspace_dist"] = float(dist.item())

    return metrics


def create_tucker_bcd_model(
    feature_dims: Sequence[int],
    feature_ranks: Sequence[int],
    tau: float = 0.0,
    ldas: Optional[Sequence[float]] = None,
    thetas: Optional[Sequence[float]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    regression_type: str = "logistic",
    fit_intercept: bool = True,
    init_with_hosvd_of_grad: bool = True,
    init_with_lbfgs_full_glm: bool = True,
    lbfgs_config: Optional[Dict[str, Any]] = None,
    rada_cfg: Optional[Dict[str, Any]] = None,
    min_gradient_norm: float = 1e-5,
    max_it: int = 150,
    verbosity: int = 0,
    log_verbosity: int = 1,
    wandb_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> TuckerRegressorBCD:
    """Instantiate a TuckerRegressorBCD (v5) model with specified hyper-parameters."""
    N = len(feature_dims)
    ldas = list(ldas) if ldas is not None else [0.0] * N
    thetas = list(thetas) if thetas is not None else [0.0] * N

    manifolds = [
        Steifel(n=dim, p=rank, device=device, dtype=dtype)
        for dim, rank in zip(feature_dims, feature_ranks)
    ]
    tregs = TuckerRegressor(
        regression_type=regression_type,
        feature_dims=feature_dims,
        feature_ranks=feature_ranks,
        task_dims=(1,),
        task_ranks=(1,),
        feature_manifolds=manifolds,
        seed=seed,
    ).to(device=device, dtype=dtype)

    subproblem_solvers = {}
    for mode in range(1, N + 1):
        lda = ldas[mode - 1]
        fd = feature_dims[mode - 1]
        fr = feature_ranks[mode - 1]
        cfg = deepcopy(rada_cfg) if rada_cfg is not None else {}
        cfg["R"] = float(lda * (fd * fr) ** 0.5) if lda != 0 else 0.0
        if "beta1" not in cfg:
            cfg["beta1"] = 0.1 * fd * (fr ** 0.5)
        if "verbosity" not in cfg:
            cfg["verbosity"] = max(0, verbosity - 1)
        subproblem_solvers[f"U_{mode}"] = RADA_RGD(**cfg)

    lbfgs_cfg = (
        dict(lbfgs_config)
        if lbfgs_config is not None
        else {
            "max_iter": 300,
            "tolerance_grad": 1e-7,
            "tolerance_change": 1e-9,
            "line_search_fn": "strong_wolfe",
        }
    )

    model = TuckerRegressorBCD(
        tucker_regressor=tregs,
        tau=float(tau),
        ldas=[float(l) for l in ldas],
        thetas=[float(t) for t in thetas],
        Ds=None,
        Ls=[None] * N,
        min_gradient_norm=min_gradient_norm,
        max_it=max_it,
        device=device,
        subproblem_solvers=subproblem_solvers,
        lbfgs_config=lbfgs_cfg,
        verbosity=max(0, verbosity - 1),
        log_verbosity=log_verbosity,
        dataloader_cfg=None,
        init_with_hosvd_of_grad=init_with_hosvd_of_grad,
        init_with_lbfgs_full_glm=init_with_lbfgs_full_glm,
        transform_covariates=True,
        wandb_kwargs=wandb_kwargs,
        wandb_kwarg=wandb_kwargs,
    )
    return model


class BaseTuckerBCDOptunaSearcher:
    """Base class for Optuna-driven hyper-parameter search on TuckerRegressorBCD."""

    def __init__(
        self,
        model_name: str,
        feature_dims: Sequence[int],
        n_trials: int = 30,
        metric: str = "val_au_roc",
        direction: str = "maximize",
        cross_validate: bool = False,
        val_ratio: float = 0.2,
        k_fold: int = 5,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        sampler_name: str = "tpe",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        regression_type: str = "logistic",
        lbfgs_config: Optional[Dict[str, Any]] = None,
        rada_cfg: Optional[Dict[str, Any]] = None,
        min_gradient_norm: float = 1e-5,
        max_it: int = 150,
        verbosity: int = 1,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.feature_dims = tuple(feature_dims)
        self.N = len(self.feature_dims)
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.cross_validate = cross_validate
        self.val_ratio = val_ratio
        self.k_fold = k_fold
        self.storage = storage
        self.study_name = study_name
        self.sampler_name = sampler_name
        self.device = device
        self.dtype = dtype
        self.regression_type = regression_type
        self.lbfgs_config = lbfgs_config
        self.rada_cfg = rada_cfg
        self.min_gradient_norm = min_gradient_norm
        self.max_it = max_it
        self.verbosity = verbosity
        self.seed = seed

        self.study: Optional[optuna.Study] = None
        self.best_model: Optional[TuckerRegressorBCD] = None
        self.best_params: Dict[str, Any] = {}
        self.summary_df: Optional[pd.DataFrame] = None

    def _sample_hparams(
        self, trial: optuna.Trial
    ) -> Tuple[Tuple[int, ...], float, List[float]]:
        """Sample (ranks, tau, ldas) for the given trial. Must be overridden by subclasses."""
        raise NotImplementedError

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        if self.sampler_name.lower() == "random":
            return optuna.samplers.RandomSampler(seed=self.seed)
        return optuna.samplers.TPESampler(seed=self.seed)

    def _create_study(self, study_name: str, storage: Optional[str]) -> optuna.Study:
        sampler = self._create_sampler()
        if storage:
            # Check if study exists
            try:
                existing_studies = optuna.study.get_all_study_names(storage=storage)
                if study_name in existing_studies:
                    optuna.delete_study(study_name=study_name, storage=storage)
            except Exception:
                pass

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction=self.direction,
            load_if_exists=True,
        )
        study.set_metric_names([self.metric])
        return study

    def search(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        B_gt: Optional[torch.Tensor] = None,
        Us_gt: Optional[Sequence[torch.Tensor]] = None,
        control_vars: Optional[Dict[str, Any]] = None,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, TuckerRegressorBCD, Dict[str, Any]]:
        """Run the Optuna optimization study and evaluate the best retrained model."""
        control_vars = dict(control_vars) if control_vars is not None else {}
        study_name = self.study_name or f"{self.model_name}-s{self.seed}"
        self.study = self._create_study(study_name=study_name, storage=self.storage)

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        n_samples = X_np.shape[0]

        # Prepare splits
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if X_val is not None and y_val is not None:
            # Fixed external validation set
            splits = [(np.arange(n_samples), np.arange(len(y_val)))]
            fixed_val = True
        elif not self.cross_validate:
            # Stratified hold-out split
            fixed_val = False
            y_flat = y_np.ravel().astype(int)
            if len(np.unique(y_flat)) > 1:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.val_ratio, random_state=self.seed
                )
                tr_idx, v_idx = next(sss.split(X_np, y_flat))
            else:
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(n_samples)
                nv = max(1, int(n_samples * self.val_ratio))
                tr_idx, v_idx = perm[nv:], perm[:nv]
            splits = [(tr_idx, v_idx)]
        else:
            # K-Fold CV
            fixed_val = False
            y_flat = y_np.ravel().astype(int)
            splitter = (
                KFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
                if len(np.unique(y_flat)) <= 1
                else None
            )
            if splitter is not None:
                for tr_idx, v_idx in splitter.split(X_np):
                    splits.append((tr_idx, v_idx))
            else:
                from sklearn.model_selection import StratifiedKFold

                skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
                for tr_idx, v_idx in skf.split(X_np, y_flat):
                    splits.append((tr_idx, v_idx))

        if self.verbosity >= 1:
            print(
                f"\n[{self.model_name}] Starting Optuna study: {study_name} "
                f"({self.n_trials} trials, metric: {self.metric}, device: {self.device})"
            )

        t_start = perf_counter()

        def objective(trial: optuna.Trial) -> float:
            ranks, tau, ldas = self._sample_hparams(trial)

            fold_val_aurocs = []
            fold_val_auprcs = []
            fold_val_accs = []
            fold_val_f1s = []
            fold_val_losses = []
            fold_val_scores = []

            fold_tr_aurocs = []
            fold_tr_auprcs = []
            fold_tr_accs = []
            fold_tr_f1s = []
            fold_tr_losses = []
            fold_tr_scores = []

            for s_idx, (tr_idx, v_idx) in enumerate(splits):
                if fixed_val:
                    X_tr_t = torch.as_tensor(X, device=self.device, dtype=self.dtype)
                    y_tr_t = torch.as_tensor(y, device=self.device, dtype=self.dtype).reshape(-1, 1)
                    X_v_t = torch.as_tensor(X_val, device=self.device, dtype=self.dtype)
                    y_v_t = torch.as_tensor(y_val, device=self.device, dtype=self.dtype).reshape(-1, 1)
                else:
                    X_tr_t = torch.as_tensor(X_np[tr_idx], device=self.device, dtype=self.dtype)
                    y_tr_t = torch.as_tensor(y_np[tr_idx], device=self.device, dtype=self.dtype).reshape(-1, 1)
                    X_v_t = torch.as_tensor(X_np[v_idx], device=self.device, dtype=self.dtype)
                    y_v_t = torch.as_tensor(y_np[v_idx], device=self.device, dtype=self.dtype).reshape(-1, 1)

                trial_run = None
                try:
                    trial_wandb_kwargs = None
                    bcd_wandb_kwargs = None
                    if wandb_kwargs is not None:
                        trial_wandb_kwargs = deepcopy(wandb_kwargs)
                        trial_tags = list(trial_wandb_kwargs.get("tags", []))
                        if "trial_run" not in trial_tags:
                            trial_tags.append("trial_run")
                        trial_wandb_kwargs["tags"] = trial_tags
                        if "project" not in trial_wandb_kwargs:
                            trial_wandb_kwargs["project"] = "tregs_bcd_synth_exp_v5"
                        base_name = trial_wandb_kwargs.get("name", self.model_name)
                        split_suffix = f"-s{s_idx}" if len(splits) > 1 else ""
                        trial_run_name = f"{base_name}-{self.model_name}-t{trial.number}{split_suffix}"
                        trial_wandb_kwargs["name"] = trial_run_name
                        trial_wandb_kwargs["config"] = {
                            **trial_wandb_kwargs.get("config", {}),
                            **control_vars,
                            "model": self.model_name,
                            "trial_number": trial.number,
                            "split_index": s_idx,
                            "feature_ranks": list(ranks),
                            **{f"rank_{m+1}": int(ranks[m]) for m in range(self.N)},
                            "tau": float(tau),
                            "ldas": list(ldas),
                            **{f"lda_{m+1}": float(ldas[m]) for m in range(self.N)},
                        }
                        trial_wandb_kwargs["reinit"] = "create_new"

                        # Pass a wandb_kwarg dict to TuckerRegressorBCD model during initialization
                        bcd_wandb_kwargs = deepcopy(trial_wandb_kwargs)
                        bcd_tags = list(bcd_wandb_kwargs.get("tags", []))
                        if "lv3" not in bcd_tags:
                            bcd_tags.append("lv3")
                        bcd_wandb_kwargs["tags"] = bcd_tags
                        bcd_wandb_kwargs["name"] = f"{trial_run_name}_lv3"
                        bcd_wandb_kwargs["reinit"] = "create_new"

                    model = create_tucker_bcd_model(
                        feature_dims=self.feature_dims,
                        feature_ranks=ranks,
                        tau=tau,
                        ldas=ldas,
                        device=self.device,
                        dtype=self.dtype,
                        regression_type=self.regression_type,
                        min_gradient_norm=self.min_gradient_norm,
                        max_it=self.max_it,
                        lbfgs_config=self.lbfgs_config,
                        rada_cfg=self.rada_cfg,
                        verbosity=0,
                        log_verbosity=1,
                        wandb_kwargs=bcd_wandb_kwargs,
                        seed=self.seed,
                    )

                    model.fit(
                        X_tr_t,
                        y_tr_t,
                        X_val=X_v_t,
                        y_val=y_v_t,
                        seed=self.seed,
                    )

                    prob_tr = model.predict(X_tr_t)
                    tr_cls = compute_classification_metrics(y_tr_t, prob_tr)
                    tr_score = float(model._score(prob_tr, y_tr_t))

                    prob_v = model.predict(X_v_t)
                    val_cls = compute_classification_metrics(y_v_t, prob_v)
                    val_score = float(model._score(prob_v, y_v_t))

                    fold_val_aurocs.append(val_cls["au_roc"])
                    fold_val_auprcs.append(val_cls["au_prc"])
                    fold_val_accs.append(val_cls["accuracy"])
                    fold_val_f1s.append(val_cls["f1"])
                    fold_val_losses.append(val_cls["loss"])
                    fold_val_scores.append(val_score)

                    fold_tr_aurocs.append(tr_cls["au_roc"])
                    fold_tr_auprcs.append(tr_cls["au_prc"])
                    fold_tr_accs.append(tr_cls["accuracy"])
                    fold_tr_f1s.append(tr_cls["f1"])
                    fold_tr_losses.append(tr_cls["loss"])
                    fold_tr_scores.append(tr_score)

                    if trial_wandb_kwargs is not None:
                        trial_run = wandb.init(**trial_wandb_kwargs)
                        trial_metrics = {
                            "val_au_roc": val_cls["au_roc"],
                            "val_au_prc": val_cls["au_prc"],
                            "val_accuracy": val_cls["accuracy"],
                            "val_balanced_accuracy": val_cls["balanced_accuracy"],
                            "val_f1": val_cls["f1"],
                            "val_precision": val_cls["precision"],
                            "val_recall": val_cls["recall"],
                            "val_loss": val_cls["loss"],
                            "val_score": val_score,
                            "train_au_roc": tr_cls["au_roc"],
                            "train_au_prc": tr_cls["au_prc"],
                            "train_accuracy": tr_cls["accuracy"],
                            "train_balanced_accuracy": tr_cls["balanced_accuracy"],
                            "train_f1": tr_cls["f1"],
                            "train_precision": tr_cls["precision"],
                            "train_recall": tr_cls["recall"],
                            "train_loss": tr_cls["loss"],
                            "train_score": tr_score,
                            "tau": float(tau),
                            **{f"rank_{m+1}": int(ranks[m]) for m in range(self.N)},
                            **{f"lda_{m+1}": float(ldas[m]) for m in range(self.N)},
                            "trial_number": trial.number,
                            "split_index": s_idx,
                        }
                        if B_gt is not None:
                            gt_m = evaluate_gt_recovery(model, B_gt=B_gt, Us_gt=Us_gt)
                            trial_metrics.update(gt_m)

                        trial_run.log(trial_metrics)
                        for k, v in trial_metrics.items():
                            if not isinstance(v, (list, tuple, dict)) and not np.isnan(v):
                                trial_run.summary[k] = v
                        trial_run.finish()
                        trial_run = None

                except Exception as e:
                    if trial_run is not None:
                        trial_run.summary["status"] = "failed"
                        trial_run.summary["error"] = str(e)
                        trial_run.finish()
                        trial_run = None
                    if self.verbosity >= 1:
                        print(f"[{self.model_name}] Trial {trial.number} split {s_idx} failed: {e}")
                    return float("-inf")

            mean_val_auroc = float(np.nanmean(fold_val_aurocs)) if fold_val_aurocs else float("nan")
            mean_val_auprc = float(np.nanmean(fold_val_auprcs)) if fold_val_auprcs else float("nan")
            mean_val_acc = float(np.nanmean(fold_val_accs)) if fold_val_accs else float("nan")
            mean_val_f1 = float(np.nanmean(fold_val_f1s)) if fold_val_f1s else float("nan")
            mean_val_loss = float(np.nanmean(fold_val_losses)) if fold_val_losses else float("nan")
            mean_val_score = float(np.nanmean(fold_val_scores)) if fold_val_scores else float("nan")

            mean_tr_auroc = float(np.nanmean(fold_tr_aurocs)) if fold_tr_aurocs else float("nan")
            mean_tr_auprc = float(np.nanmean(fold_tr_auprcs)) if fold_tr_auprcs else float("nan")
            mean_tr_acc = float(np.nanmean(fold_tr_accs)) if fold_tr_accs else float("nan")
            mean_tr_f1 = float(np.nanmean(fold_tr_f1s)) if fold_tr_f1s else float("nan")
            mean_tr_loss = float(np.nanmean(fold_tr_losses)) if fold_tr_losses else float("nan")
            mean_tr_score = float(np.nanmean(fold_tr_scores)) if fold_tr_scores else float("nan")

            trial.set_user_attr("val_au_roc", mean_val_auroc)
            trial.set_user_attr("val_au_prc", mean_val_auprc)
            trial.set_user_attr("val_accuracy", mean_val_acc)
            trial.set_user_attr("val_f1", mean_val_f1)
            trial.set_user_attr("val_loss", mean_val_loss)
            trial.set_user_attr("val_score", mean_val_score)

            trial.set_user_attr("train_au_roc", mean_tr_auroc)
            trial.set_user_attr("train_au_prc", mean_tr_auprc)
            trial.set_user_attr("train_accuracy", mean_tr_acc)
            trial.set_user_attr("train_f1", mean_tr_f1)
            trial.set_user_attr("train_loss", mean_tr_loss)
            trial.set_user_attr("train_score", mean_tr_score)

            trial.set_user_attr("feature_ranks", list(ranks))
            trial.set_user_attr("tau", float(tau))
            trial.set_user_attr("ldas", list(ldas))

            if np.isnan(mean_val_auroc):
                return mean_val_score
            return mean_val_auroc

        optuna.logging.set_verbosity(
            optuna.logging.WARNING if self.verbosity < 2 else optuna.logging.INFO
        )
        self.study.optimize(objective, n_trials=self.n_trials)
        elapsed_total = perf_counter() - t_start

        best_trial = self.study.best_trial
        self.best_params = best_trial.params

        # Reconstruct best ranks, tau, ldas from best_trial
        best_ranks = tuple(
            best_trial.params[f"rank_{m+1}"]
            if f"rank_{m+1}" in best_trial.params
            else best_trial.user_attrs.get("feature_ranks", [3] * self.N)[m]
            for m in range(self.N)
        )
        best_tau = float(
            best_trial.params.get("tau", best_trial.user_attrs.get("tau", 0.0))
        )
        best_ldas = [
            float(
                best_trial.params[f"lda_{m+1}"]
                if f"lda_{m+1}" in best_trial.params
                else best_trial.user_attrs.get("ldas", [0.0] * self.N)[m]
            )
            for m in range(self.N)
        ]

        if self.verbosity >= 1:
            print(
                f"[{self.model_name}] Completed {len(self.study.trials)} trials in {elapsed_total:.1f}s. "
                f"Best Trial #{best_trial.number}: val_au_roc={best_trial.value:.4f}, "
                f"ranks={best_ranks}, tau={best_tau:g}, ldas={best_ldas}"
            )

        # Retrain best model on full training set
        X_full_t = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        y_full_t = torch.as_tensor(y, device=self.device, dtype=self.dtype).reshape(-1, 1)
        X_v_full = (
            torch.as_tensor(X_val, device=self.device, dtype=self.dtype)
            if X_val is not None
            else None
        )
        y_v_full = (
            torch.as_tensor(y_val, device=self.device, dtype=self.dtype).reshape(-1, 1)
            if y_val is not None
            else None
        )

        retrain_wandb_kwargs = None
        best_bcd_wandb_kwargs = None
        if wandb_kwargs is not None:
            retrain_wandb_kwargs = deepcopy(wandb_kwargs)
            retrain_tags = list(retrain_wandb_kwargs.get("tags", []))
            if "best_model" not in retrain_tags:
                retrain_tags.append("best_model")
            retrain_wandb_kwargs["tags"] = retrain_tags
            if "project" not in retrain_wandb_kwargs:
                retrain_wandb_kwargs["project"] = "tregs_bcd_synth_exp_v5"
            base_name = retrain_wandb_kwargs.get("name", self.model_name)
            retrain_run_name = f"{base_name}-{self.model_name}-best_retrained"
            retrain_wandb_kwargs["name"] = retrain_run_name
            retrain_wandb_kwargs["config"] = {
                **retrain_wandb_kwargs.get("config", {}),
                **control_vars,
                "model": self.model_name,
                "feature_ranks": list(best_ranks),
                **{f"rank_{m+1}": int(best_ranks[m]) for m in range(self.N)},
                "tau": float(best_tau),
                "ldas": list(best_ldas),
                **{f"lda_{m+1}": float(best_ldas[m]) for m in range(self.N)},
            }
            retrain_wandb_kwargs["reinit"] = "create_new"

            # Pass a wandb_kwarg dict to TuckerRegressorBCD during initialization
            best_bcd_wandb_kwargs = deepcopy(retrain_wandb_kwargs)
            best_tags = list(best_bcd_wandb_kwargs.get("tags", []))
            if "lv3" not in best_tags:
                best_tags.append("lv3")
            best_bcd_wandb_kwargs["tags"] = best_tags
            best_bcd_wandb_kwargs["name"] = f"{retrain_run_name}_lv3"
            best_bcd_wandb_kwargs["reinit"] = "create_new"

        self.best_model = create_tucker_bcd_model(
            feature_dims=self.feature_dims,
            feature_ranks=best_ranks,
            tau=best_tau,
            ldas=best_ldas,
            device=self.device,
            dtype=self.dtype,
            regression_type=self.regression_type,
            min_gradient_norm=self.min_gradient_norm,
            max_it=self.max_it,
            lbfgs_config=self.lbfgs_config,
            rada_cfg=self.rada_cfg,
            verbosity=max(0, self.verbosity - 1),
            log_verbosity=1,
            wandb_kwargs=best_bcd_wandb_kwargs,
            seed=self.seed,
        )

        self.best_model.fit(
            X_full_t,
            y_full_t,
            X_val=X_v_full,
            y_val=y_v_full,
            seed=self.seed,
        )

        # Evaluate full training set
        prob_tr_full = self.best_model.predict(X_full_t)
        tr_cls = compute_classification_metrics(y_full_t, prob_tr_full)
        tr_score = float(self.best_model._score(prob_tr_full, y_full_t))

        retrain_metrics = {
            "train_au_roc": tr_cls["au_roc"],
            "train_au_prc": tr_cls["au_prc"],
            "train_accuracy": tr_cls["accuracy"],
            "train_balanced_accuracy": tr_cls["balanced_accuracy"],
            "train_f1": tr_cls["f1"],
            "train_precision": tr_cls["precision"],
            "train_recall": tr_cls["recall"],
            "train_loss": tr_cls["loss"],
            "train_score": tr_score,
            "tau": float(best_tau),
            **{f"rank_{m+1}": int(best_ranks[m]) for m in range(self.N)},
            **{f"lda_{m+1}": float(best_ldas[m]) for m in range(self.N)},
        }

        # Evaluate validation set if present
        val_cls = {}
        if X_v_full is not None and y_v_full is not None:
            prob_v_full = self.best_model.predict(X_v_full)
            val_cls = compute_classification_metrics(y_v_full, prob_v_full)
            val_score = float(self.best_model._score(prob_v_full, y_v_full))
            retrain_metrics.update({
                "val_au_roc": val_cls["au_roc"],
                "val_au_prc": val_cls["au_prc"],
                "val_accuracy": val_cls["accuracy"],
                "val_balanced_accuracy": val_cls["balanced_accuracy"],
                "val_f1": val_cls["f1"],
                "val_precision": val_cls["precision"],
                "val_recall": val_cls["recall"],
                "val_loss": val_cls["loss"],
                "val_score": val_score,
            })

        # Out-of-sample test evaluation
        test_metrics: Dict[str, float] = {}
        test_cls = {}
        if X_test is not None and y_test is not None:
            X_test_t = torch.as_tensor(X_test, device=self.device, dtype=self.dtype)
            y_test_t = torch.as_tensor(y_test, device=self.device, dtype=self.dtype).reshape(-1, 1)
            prob_test = self.best_model.predict(X_test_t)
            test_cls = compute_classification_metrics(y_test_t, prob_test)
            test_score = float(self.best_model._score(prob_test, y_test_t))
            test_metrics = {
                "test_score": test_score,
                "test_loss": test_cls["loss"],
                "test_au_roc": test_cls["au_roc"],
                "test_au_prc": test_cls["au_prc"],
                "test_accuracy": test_cls["accuracy"],
                "test_balanced_accuracy": test_cls["balanced_accuracy"],
                "test_f1": test_cls["f1"],
                "test_precision": test_cls["precision"],
                "test_recall": test_cls["recall"],
            }
            retrain_metrics.update(test_metrics)

        gt_metrics = evaluate_gt_recovery(self.best_model, B_gt=B_gt, Us_gt=Us_gt)
        retrain_metrics.update(gt_metrics)

        if retrain_wandb_kwargs is not None:
            retrain_run = wandb.init(**retrain_wandb_kwargs)
            retrain_run.log(retrain_metrics)
            for k, v in retrain_metrics.items():
                if not isinstance(v, (list, tuple, dict)) and not np.isnan(v):
                    retrain_run.summary[k] = v
            retrain_run.finish()

        # Compile row
        summary_row: Dict[str, Any] = {
            "Model": self.model_name,
            "tau": float(best_tau),
            **{f"lda_{m+1}": float(best_ldas[m]) for m in range(self.N)},
            "best_C": np.nan,
            **control_vars,
            **test_metrics,
            "mean_val_score": best_trial.user_attrs.get("val_score", np.nan),
            "std_val_score": 0.0,
            "mean_val_loss": best_trial.user_attrs.get("val_loss", np.nan),
            "std_val_loss": 0.0,
            "mean_val_auroc": best_trial.user_attrs.get("val_au_roc", float(best_trial.value)),
            "mean_val_auprc": best_trial.user_attrs.get("val_au_prc", np.nan),
            "mean_val_accuracy": best_trial.user_attrs.get("val_accuracy", np.nan),
            "mean_val_f1": best_trial.user_attrs.get("val_f1", np.nan),
            "mean_train_score": best_trial.user_attrs.get("train_score", np.nan),
            "std_train_score": 0.0,
            "mean_train_loss": best_trial.user_attrs.get("train_loss", np.nan),
            "std_train_loss": 0.0,
            "mean_train_auroc": best_trial.user_attrs.get("train_au_roc", np.nan),
            "mean_train_auprc": best_trial.user_attrs.get("train_au_prc", np.nan),
            "mean_train_accuracy": best_trial.user_attrs.get("train_accuracy", np.nan),
            "mean_train_f1": best_trial.user_attrs.get("train_f1", np.nan),
            "time": float(elapsed_total),
            **gt_metrics,
            **{f"rank_{m+1}": int(best_ranks[m]) for m in range(self.N)},
        }

        self.summary_df = pd.DataFrame([summary_row])

        # Save study trials to CSV locally
        if self.storage and self.storage.startswith("sqlite:///"):
            db_path = Path(self.storage.replace("sqlite:///", ""))
            trials_csv = db_path.parent / f"{study_name}_trials.csv"
            try:
                trials_df = self.study.trials_dataframe()
                trials_df.to_csv(trials_csv, index=False)
            except Exception:
                pass

        return self.summary_df, self.best_model, best_trial.params


class TuckerRegsSearcher(BaseTuckerBCDOptunaSearcher):
    """Searches for best feature ranks in TuckerRegressorBCD with fixed tau=0, ldas=0."""

    def __init__(
        self,
        ranks_ranges: Sequence[Union[Tuple[int, int], Sequence[int]]],
        **kwargs,
    ):
        kwargs["model_name"] = kwargs.get("model_name", "TuckerRegs")
        super().__init__(**kwargs)
        self.ranks_ranges = list(ranks_ranges)

    def _sample_hparams(
        self, trial: optuna.Trial
    ) -> Tuple[Tuple[int, ...], float, List[float]]:
        ranks = []
        for m in range(self.N):
            r_spec = self.ranks_ranges[m]
            if isinstance(r_spec, (tuple, list)) and len(r_spec) == 2:
                r_m = trial.suggest_int(f"rank_{m+1}", int(r_spec[0]), int(r_spec[1]))
            else:
                r_m = trial.suggest_categorical(f"rank_{m+1}", [int(x) for x in r_spec])
            ranks.append(r_m)

        tau = 0.0
        ldas = [0.0] * self.N
        return tuple(ranks), tau, ldas


class RidgeTuckerRegsSearcher(BaseTuckerBCDOptunaSearcher):
    """Searches for best feature ranks and core ridge penalty tau with fixed ldas=0."""

    def __init__(
        self,
        ranks_ranges: Sequence[Union[Tuple[int, int], Sequence[int]]],
        tau_range: Union[Tuple[float, float], Sequence[float]] = (1e-5, 1.0),
        **kwargs,
    ):
        kwargs["model_name"] = kwargs.get("model_name", "RidgeTuckerRegs")
        super().__init__(**kwargs)
        self.ranks_ranges = list(ranks_ranges)
        self.tau_range = tau_range

    def _sample_hparams(
        self, trial: optuna.Trial
    ) -> Tuple[Tuple[int, ...], float, List[float]]:
        ranks = []
        for m in range(self.N):
            r_spec = self.ranks_ranges[m]
            if isinstance(r_spec, (tuple, list)) and len(r_spec) == 2:
                r_m = trial.suggest_int(f"rank_{m+1}", int(r_spec[0]), int(r_spec[1]))
            else:
                r_m = trial.suggest_categorical(f"rank_{m+1}", [int(x) for x in r_spec])
            ranks.append(r_m)

        if isinstance(self.tau_range, (tuple, list)) and len(self.tau_range) == 2:
            tau = trial.suggest_float(
                "tau", float(self.tau_range[0]), float(self.tau_range[1]), log=True
            )
        else:
            tau = trial.suggest_categorical("tau", [float(t) for t in self.tau_range])

        ldas = [0.0] * self.N
        return tuple(ranks), tau, ldas


class SparseRTuckerRegsSearcher(BaseTuckerBCDOptunaSearcher):
    """Searches for core ridge penalty tau and factor sparsity penalties ldas.

    Uses fixed feature ranks inherited from RidgeTuckerRegs.
    """

    def __init__(
        self,
        feature_ranks: Sequence[int],
        tau_range: Union[Tuple[float, float], Sequence[float]] = (1e-5, 1.0),
        ldas_ranges: Optional[
            Sequence[Union[Tuple[float, float], Sequence[float], float, None]]
        ] = None,
        **kwargs,
    ):
        kwargs["model_name"] = kwargs.get("model_name", "SparseRTuckerRegs")
        super().__init__(**kwargs)
        self.feature_ranks = tuple(int(r) for r in feature_ranks)
        self.tau_range = tau_range
        if ldas_ranges is not None:
            self.ldas_ranges = list(ldas_ranges)
        elif self.N == 3:
            self.ldas_ranges = [(1e-6, 0.5), (1e-6, 0.5), 0.0]
        else:
            self.ldas_ranges = [(1e-6, 0.5)] * self.N

    def _sample_hparams(
        self, trial: optuna.Trial
    ) -> Tuple[Tuple[int, ...], float, List[float]]:
        ranks = self.feature_ranks

        # Sample tau
        if isinstance(self.tau_range, (tuple, list)) and len(self.tau_range) == 2:
            tau = trial.suggest_float(
                "tau", float(self.tau_range[0]), float(self.tau_range[1]), log=True
            )
        else:
            tau = trial.suggest_categorical("tau", [float(t) for t in self.tau_range])

        # Sample ldas
        ldas = []
        for m in range(self.N):
            l_spec = self.ldas_ranges[m] if m < len(self.ldas_ranges) else None
            if l_spec is None:
                lda_m = 0.0
            elif isinstance(l_spec, (int, float)):
                lda_m = float(l_spec)
            elif isinstance(l_spec, (tuple, list)) and len(l_spec) == 2:
                lda_m = trial.suggest_float(
                    f"lda_{m+1}", float(l_spec[0]), float(l_spec[1]), log=True
                )
            else:
                lda_m = trial.suggest_categorical(
                    f"lda_{m+1}", [float(x) for x in l_spec]
                )
            ldas.append(lda_m)

        return ranks, tau, ldas


def run_tucker_optuna_suite(
    models_cfg: Dict[str, Any],
    feature_dims: Sequence[int],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    X_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    B_gt: Optional[torch.Tensor] = None,
    Us_gt: Optional[Sequence[torch.Tensor]] = None,
    control_vars: Optional[Dict[str, Any]] = None,
    wandb_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    optuna_storage_dir: Optional[Union[str, Path]] = None,
    lbfgs_config: Optional[Dict[str, Any]] = None,
    rada_cfg: Optional[Dict[str, Any]] = None,
    max_it: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the 3-model Tucker suite (TuckerRegs -> RidgeTuckerRegs -> SparseRTuckerRegs).

    Chains optimal ranks discovered in RidgeTuckerRegs into SparseRTuckerRegs.
    Logs each trial to local Optuna storage and summary table to WandB.
    """
    control_vars = dict(control_vars) if control_vars is not None else {}
    dfs: List[pd.DataFrame] = []
    optuna_dir = Path(optuna_storage_dir or "results/optuna_studies")
    optuna_dir.mkdir(parents=True, exist_ok=True)

    # 1. TuckerRegs
    if "TuckerRegs" in models_cfg:
        m_cfg = models_cfg["TuckerRegs"]
        opt_cfg = m_cfg.get("optuna_cfg", {})
        study_name = f"TuckerRegs-s{seed}-n{control_vars.get('n_train', 0)}"
        storage_url = f"sqlite:///{optuna_dir / study_name}.db"

        tucker_searcher = TuckerRegsSearcher(
            ranks_ranges=opt_cfg.get("ranks_ranges", [(1, 5)] * len(feature_dims)),
            feature_dims=feature_dims,
            n_trials=opt_cfg.get("n_trials", 30),
            metric=opt_cfg.get("metric", "val_au_roc"),
            direction=opt_cfg.get("direction", "maximize"),
            cross_validate=opt_cfg.get("cross_validate", False),
            val_ratio=opt_cfg.get("val_ratio", 0.2),
            k_fold=opt_cfg.get("k_fold", 5),
            storage=storage_url,
            study_name=study_name,
            device=device,
            lbfgs_config=lbfgs_config,
            rada_cfg=rada_cfg,
            max_it=opt_cfg.get("max_it", max_it if max_it is not None else 150),
            seed=seed,
            verbosity=opt_cfg.get("verbosity", 1),
        )
        t_df, _, _ = tucker_searcher.search(
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
        )
        dfs.append(t_df)

    # 2. RidgeTuckerRegs
    best_ranks_for_sparse = tuple([3] * len(feature_dims))
    if "RidgeTuckerRegs" in models_cfg:
        m_cfg = models_cfg["RidgeTuckerRegs"]
        opt_cfg = m_cfg.get("optuna_cfg", {})
        study_name = f"RidgeTuckerRegs-s{seed}-n{control_vars.get('n_train', 0)}"
        storage_url = f"sqlite:///{optuna_dir / study_name}.db"

        ridge_searcher = RidgeTuckerRegsSearcher(
            ranks_ranges=opt_cfg.get("ranks_ranges", [(1, 5)] * len(feature_dims)),
            tau_range=opt_cfg.get("tau_range", (1e-5, 1.0)),
            feature_dims=feature_dims,
            n_trials=opt_cfg.get("n_trials", 30),
            metric=opt_cfg.get("metric", "val_au_roc"),
            direction=opt_cfg.get("direction", "maximize"),
            cross_validate=opt_cfg.get("cross_validate", False),
            val_ratio=opt_cfg.get("val_ratio", 0.2),
            k_fold=opt_cfg.get("k_fold", 5),
            storage=storage_url,
            study_name=study_name,
            device=device,
            lbfgs_config=lbfgs_config,
            rada_cfg=rada_cfg,
            max_it=opt_cfg.get("max_it", max_it if max_it is not None else 150),
            seed=seed,
            verbosity=opt_cfg.get("verbosity", 1),
        )
        r_df, _, r_params = ridge_searcher.search(
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
        )
        dfs.append(r_df)

        best_ranks_for_sparse = tuple(
            int(r_params[f"rank_{m+1}"]) for m in range(len(feature_dims))
        )
        if opt_cfg.get("verbosity", 1) >= 1:
            print(f"[RunSuite] Passing best Ridge ranks {best_ranks_for_sparse} to SparseRTuckerRegs.")

    # 3. SparseRTuckerRegs (inherits best_ranks from RidgeTuckerRegs)
    if "SparseRTuckerRegs" in models_cfg:
        m_cfg = models_cfg["SparseRTuckerRegs"]
        opt_cfg = m_cfg.get("optuna_cfg", {})
        study_name = f"SparseRTuckerRegs-s{seed}-n{control_vars.get('n_train', 0)}"
        storage_url = f"sqlite:///{optuna_dir / study_name}.db"

        sparse_searcher = SparseRTuckerRegsSearcher(
            feature_ranks=best_ranks_for_sparse,
            tau_range=opt_cfg.get("tau_range", (1e-5, 1.0)),
            ldas_ranges=opt_cfg.get(
                "ldas_ranges", [(1e-6, 0.5), (1e-6, 0.5), 0.0]
            ),
            feature_dims=feature_dims,
            n_trials=opt_cfg.get("n_trials", 30),
            metric=opt_cfg.get("metric", "val_au_roc"),
            direction=opt_cfg.get("direction", "maximize"),
            cross_validate=opt_cfg.get("cross_validate", False),
            val_ratio=opt_cfg.get("val_ratio", 0.2),
            k_fold=opt_cfg.get("k_fold", 5),
            storage=storage_url,
            study_name=study_name,
            device=device,
            lbfgs_config=lbfgs_config,
            rada_cfg=rada_cfg,
            max_it=opt_cfg.get("max_it", max_it if max_it is not None else 150),
            seed=seed,
            verbosity=opt_cfg.get("verbosity", 1),
        )
        s_df, _, _ = sparse_searcher.search(
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
        )
        dfs.append(s_df)

    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Log summary table to WandB
    if wandb_kwargs is not None:
        try:
            summary_kwargs = deepcopy(wandb_kwargs)
            summary_tags = list(summary_kwargs.get("tags", []))
            if "summary_table" not in summary_tags:
                summary_tags.append("summary_table")
            summary_kwargs["tags"] = summary_tags
            base_name = summary_kwargs.get("name", "experiment")
            summary_kwargs["name"] = f"{base_name}-summary"
            summary_kwargs["reinit"] = "create_new"
            s_run = wandb.init(**summary_kwargs)
            s_run.log({"summary_table": wandb.Table(dataframe=combined_df)})
            for _, row in combined_df.iterrows():
                m_name = row.get("Model", "Unknown")
                for metric_key in [
                    "test_au_roc", "test_au_prc", "test_accuracy", "test_f1", "test_loss",
                    "mean_val_auroc", "mean_val_auprc", "mean_val_accuracy", "mean_val_f1",
                    "mean_train_auroc", "mean_train_auprc", "mean_train_accuracy", "mean_train_f1",
                    "B_est_diff", "B_est_rel_diff", "subspace_error_mean", "time"
                ]:
                    if metric_key in row and pd.notna(row[metric_key]):
                        s_run.summary[f"{m_name}/{metric_key}"] = row[metric_key]
            s_run.finish()
        except Exception as e:
            print(f"Warning: WandB summary table logging failed: {e}")

    return combined_df

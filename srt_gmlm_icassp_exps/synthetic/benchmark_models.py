"""Benchmark GLM models for tensor regression synthetic experiments.

Provides:
  - Vectorized Lasso Logistic Regression (L1-penalized flat GLM)
  - Vectorized ElasticNet Logistic Regression (L1+L2-penalized flat GLM)
  - Vectorized Ridge Logistic Regression (L2-penalized flat GLM)
"""

from time import perf_counter
from typing import Optional, Sequence, Dict, Any
import numpy as np
import torch
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression


class VectorizedLogisticRegressionBenchmark:
    """Vectorized Logistic Regression baseline (L1/Lasso, L2/Ridge, ElasticNet)."""

    def __init__(
        self,
        penalty: str = 'l1',
        C_range: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 0,
    ):
        self.penalty = penalty
        self.C_range = list(C_range)
        self.l1_ratio = l1_ratio if penalty == 'elasticnet' else None
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if penalty == 'l1':
            self.solver = 'saga'
            self.model_name = 'Vectorized-Lasso'
        elif penalty == 'elasticnet':
            self.solver = 'saga'
            self.model_name = 'Vectorized-ElasticNet'
        elif penalty == 'l2':
            self.solver = 'lbfgs'
            self.model_name = 'Vectorized-Ridge'
        else:
            raise ValueError(f"Unknown penalty: {penalty}")

    def fit_and_eval(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        B_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fit over C_range on train/val and evaluate on test set."""
        start_time = perf_counter()

        feature_shape = tuple(X_train.shape[1:])
        X_tr_flat = X_train.detach().cpu().numpy().reshape(X_train.shape[0], -1)
        y_tr_flat = y_train.detach().cpu().numpy().ravel()
        X_v_flat = X_val.detach().cpu().numpy().reshape(X_val.shape[0], -1)
        y_v_flat = y_val.detach().cpu().numpy().ravel()

        best_score = -float('inf')
        best_clf = None
        best_C = self.C_range[0]

        for C in self.C_range:
            kwargs: Dict[str, Any] = {
                'penalty': self.penalty,
                'C': float(C),
                'solver': self.solver,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'random_state': self.random_state,
            }
            if self.l1_ratio is not None:
                kwargs['l1_ratio'] = self.l1_ratio

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                clf = LogisticRegression(**kwargs)
                try:
                    clf.fit(X_tr_flat, y_tr_flat)
                    val_score = clf.score(X_v_flat, y_v_flat)
                    if val_score > best_score:
                        best_score = val_score
                        best_clf = clf
                        best_C = float(C)
                except Exception:
                    continue

        if best_clf is None:
            best_clf = LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs', max_iter=self.max_iter, random_state=self.random_state
            ).fit(X_tr_flat, y_tr_flat)
            best_C = 1.0

        fit_time = perf_counter() - start_time

        train_score = float(best_clf.score(X_tr_flat, y_tr_flat))
        val_score = float(best_clf.score(X_v_flat, y_v_flat))

        prob_tr = np.clip(best_clf.predict_proba(X_tr_flat)[:, 1], 1e-12, 1 - 1e-12)
        train_loss = float(-np.mean(y_tr_flat * np.log(prob_tr) + (1 - y_tr_flat) * np.log(1 - prob_tr)))
        prob_v = np.clip(best_clf.predict_proba(X_v_flat)[:, 1], 1e-12, 1 - 1e-12)
        val_loss = float(-np.mean(y_v_flat * np.log(prob_v) + (1 - y_v_flat) * np.log(1 - prob_v)))

        test_score = np.nan
        test_loss = np.nan
        if X_test is not None and y_test is not None:
            X_te_flat = X_test.detach().cpu().numpy().reshape(X_test.shape[0], -1)
            y_te_flat = y_test.detach().cpu().numpy().ravel()
            test_score = float(best_clf.score(X_te_flat, y_te_flat))
            prob_te = np.clip(best_clf.predict_proba(X_te_flat)[:, 1], 1e-12, 1 - 1e-12)
            test_loss = float(-np.mean(y_te_flat * np.log(prob_te) + (1 - y_te_flat) * np.log(1 - prob_te)))

        coef_flat = best_clf.coef_.ravel()
        B_hat = torch.tensor(coef_flat.reshape(feature_shape), dtype=torch.float64)
        nnz_ratio = float((np.abs(coef_flat) >= 1e-5).sum() / coef_flat.size)

        metrics: Dict[str, Any] = {
            'Model': self.model_name,
            'tau': 0.0,
            'best_C': best_C,
            'mean_train_score': train_score,
            'std_train_score': 0.0,
            'mean_train_loss': train_loss,
            'std_train_loss': 0.0,
            'mean_val_score': val_score,
            'std_val_score': 0.0,
            'mean_val_loss': val_loss,
            'std_val_loss': 0.0,
            'test_score': test_score,
            'test_loss': test_loss,
            'B_nnz_ratio': nnz_ratio,
            'time': fit_time,
        }

        if B_gt is not None:
            B_gt_t = torch.as_tensor(B_gt, dtype=torch.float64).cpu()
            rel_err = float((torch.linalg.norm(B_hat - B_gt_t) / torch.linalg.norm(B_gt_t)).item())
            metrics['rel_coef_error'] = rel_err

        return metrics


def run_benchmark(
    model_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    B_gt: Optional[torch.Tensor] = None,
    control_vars: Optional[dict] = None,
    benchmark_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Unified runner for benchmark GLM models."""
    cfg = benchmark_cfg or {}

    if model_name in ('vectorized_lasso', 'lasso'):
        bm = VectorizedLogisticRegressionBenchmark(
            penalty='l1',
            C_range=cfg.get('C_range', (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)),
            max_iter=cfg.get('max_iter', 1000),
        )
    elif model_name in ('vectorized_elasticnet', 'elasticnet'):
        bm = VectorizedLogisticRegressionBenchmark(
            penalty='elasticnet',
            C_range=cfg.get('C_range', (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)),
            l1_ratio=cfg.get('l1_ratio', 0.5),
            max_iter=cfg.get('max_iter', 1000),
        )
    elif model_name in ('vectorized_ridge', 'ridge'):
        bm = VectorizedLogisticRegressionBenchmark(
            penalty='l2',
            C_range=cfg.get('C_range', (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)),
            max_iter=cfg.get('max_iter', 1000),
        )
    else:
        raise ValueError(f"Unknown benchmark model name: {model_name}")

    metrics = bm.fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, B_gt)
    if control_vars is not None:
        metrics = {**control_vars, **metrics}

    return pd.DataFrame([metrics])


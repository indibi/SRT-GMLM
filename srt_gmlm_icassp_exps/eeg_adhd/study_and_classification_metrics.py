import os
os.environ["WANDB_SILENT"] = "True"
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score, roc_auc_score,
)


from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.solvers.manifold.rada import RADA_RGD

from srt_gmlm_icassp_exps.eeg_adhd.create_log_psd_samples import (
    standardize,
    eeg_log_psd,
    load_train_val_test_split_subjects_process
)


def compute_classification_metrics(
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        regression_type='logistic',
    ) -> Dict[str, float]:
    """Compute classification accuracy, F1 score, AU-ROC, and AU-PRC.

    Handles both binary logistic regression and multiclass multinomial regression,
    with graceful handling of edge cases (e.g. single-class folds).
    """
    if regression_type not in ("logistic", "multinomial"):
        return {
            "accuracy": float("nan"),
            "f1": float("nan"),
            "au_roc": float("nan"),
            "au_prc": float("nan"),
        }

    yt = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    yp = y_prob.detach().cpu().numpy() if isinstance(y_prob, torch.Tensor) else np.asarray(y_prob)

    metrics: Dict[str, float] = {}

    if regression_type == "logistic":
        yt_flat = yt.ravel().astype(int)
        yp_flat = yp.ravel().astype(float)
        y_pred = (yp_flat >= 0.5).astype(int)

        metrics["accuracy"] = float(accuracy_score(yt_flat, y_pred))
        metrics["f1"] = float(f1_score(yt_flat, y_pred, average="binary", zero_division=0))

        unique_classes = np.unique(yt_flat)
        if len(unique_classes) > 1:
            try:
                metrics["au_roc"] = float(roc_auc_score(yt_flat, yp_flat))
            except Exception:
                metrics["au_roc"] = float("nan")
            try:
                metrics["au_prc"] = float(average_precision_score(yt_flat, yp_flat))
            except Exception:
                metrics["au_prc"] = float("nan")
        else:
            metrics["au_roc"] = float("nan")
            metrics["au_prc"] = float("nan")

    elif regression_type == "multinomial":
        yt_flat = yt.ravel().astype(int)
        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        y_pred = np.argmax(yp, axis=1)

        metrics["accuracy"] = float(accuracy_score(yt_flat, y_pred))
        metrics["f1"] = float(f1_score(yt_flat, y_pred, average="macro", zero_division=0))

        unique_classes = np.unique(yt_flat)
        if len(unique_classes) > 1:
            try:
                metrics["au_roc"] = float(roc_auc_score(yt_flat, yp, multi_class="ovr", average="macro"))
            except Exception:
                metrics["au_roc"] = float("nan")
            try:
                from sklearn.preprocessing import label_binarize

                n_classes = yp.shape[1] if yp.ndim > 1 else len(unique_classes)
                Y_bin = label_binarize(yt_flat, classes=list(range(n_classes)))
                metrics["au_prc"] = float(average_precision_score(Y_bin, yp, average="macro"))
            except Exception:
                metrics["au_prc"] = float("nan")
        else:
            metrics["au_roc"] = float("nan")
            metrics["au_prc"] = float("nan")
    return metrics





def search_hyperparams(
    study_name,
    fif_dir,
    lda1_range, lda2_range, tau_range, ranks,
    n_sample_per_subj=3, test_ratio=0.2, val_ratio=0.25, seed=43,
    max_it=30, min_grad_norm=1e-3,
    fused=False,
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    storage='sqlite:///eeg_adhd_hp_search_optuna.db',
    ):
    fsd = '_fused_' if fused else ''
    sname = study_name + fsd + str(ranks) + "_" + f'_nsps{n_sample_per_subj}'+ str(seed)
    study = optuna.create_study(
        directions=['maximize','minimize'],
        study_name=sname,
        storage=storage,
        load_if_exists=True,
    )
    dset = load_train_val_test_split_subjects_process(
        n_sample_per_subj,
        fif_dir,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        )
    X_train, _, _ = standardize(dset['X_train'])
    X_val, _, _ = standardize(dset['X_val'])
    X_test, _, _ = standardize(dset['X_test'])

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(dset['y_train']).to(device)
    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(dset['y_val']).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(dset['y_test']).to(device)

    def objective(trial):
        trial_num = trial.number
        lda1 = float(trial.suggest_float(
            'lda1', lda1_range[0], lda1_range[-1]+1e-12, log=True
            ))
        lda2 = float(trial.suggest_float(
            'lda2', lda2_range[0], lda2_range[-1]+1e-12, log= not (len(lda2_range)==1 and lda2_range[0]==0),
            ))
        tau = float(trial.suggest_float(
            'tau', tau_range[0], tau_range[-1]+1e-12, log=True
            ))
        ldas = (lda1, lda2)
        # with wandb.init(
        #     project='eeg_adhd_classification',
        #     name=sname + f"_tn{trial_num}",
        #     config={
        #         'ranks': ranks,
        #         'n_sample_per_subj': n_sample_per_subj,
        #         'fused': fused,
        #         **trial.params},
        #     reinit='create_new'
        #     ) as run:

        dtype = X_train.dtype
        tregs = TuckerRegressor(
            'logistic',
            feature_dims=X_train.shape[1:],
            feature_ranks=ranks,
            force_lr_features=False,
            task_dims=(1,),
            ).to(device=device, dtype=dtype)
        wandb_kwargs = None
            # {
            # 'project':'eeg_adhd_classification',
            # 'name': sname+'_lv3',
            # 'group': sname,
            # 'reinit': 'create_new',
            # 'tags': ['lv3']
            # }
        model = TuckerRegressorBCD(
            tregs, tau=tau, ldas=ldas,
            thetas=(0,0),
            device=device, Ls=[None,None], Ds=None,
            min_gradient_norm=min_grad_norm,
            init_with_hosvd_of_grad=False,
            init_with_lbfgs_full_glm=True,
            fit_intercept=True,
            lbfgs_config={
                'max_iter': 500,
                'tolerance_grad': 1e-8,
                'tolerance_change': 1e-9,
                'line_search_fn': 'strong_wolfe'
                },
            max_it=max_it,
            verbosity=0,
            wandb_kwargs=wandb_kwargs,
            )
        model.subproblem_solvers['U_2'].log_verbosity=0
        model.subproblem_solvers['U_1'].log_verbosity=0
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        prob_tr = model.predict(X_train)
        metrics = compute_classification_metrics(y_train, prob_tr)
        train_metrics = {'train_'+k: v for k,v in metrics.items()}

        prob_val = model.predict(X_val)
        metrics = compute_classification_metrics(y_val, prob_val)
        val_metrics = {'val_'+k: v for k,v in metrics.items()}
        
        prob_test = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, prob_test)
        test_metrics = {'test_'+k: v for k,v in metrics.items()}
        # run.log(
        #     {**train_metrics, **val_metrics, **test_metrics},
        #     step=trial_num
        #     )
        return float(val_metrics['val_accuracy']), float(model._v_loss)
    for i, tau_ in enumerate(tau_range):
        for j, lda1_ in enumerate(lda1_range):
            for k, lda2_ in enumerate(lda2_range):
                study.enqueue_trial(
                    {'tau': float(tau_),
                    'lda1': float(lda1_),
                    'lda2': float(lda2_)
                    }
                )
    n_trials = len(tau_range)*len(lda1_range)*len(lda2_range)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study



def fused_lasso_matrix_torch(
    n: int, dtype=torch.float64, device="cpu"
    ) -> torch.Tensor:
    """Constructs the (n-1) x n first-order difference matrix for 1D fused lasso

    as a sparse PyTorch COO tensor.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    m = n - 1  # Number of rows
    row_indices = torch.repeat_interleave(torch.arange(m, device=device), 2)
    col_indices = torch.stack(
        [
            torch.arange(m, device=device),
            torch.arange(1, n, device=device),
        ],
        dim=1,
    ).flatten()
    indices = torch.stack([row_indices, col_indices])
    values = (
        torch.tensor([-1.0, 1.0], dtype=dtype, device=device).repeat(m)
    )
    return torch.sparse_coo_tensor(
        indices, values, size=(m, n), dtype=dtype, device=device
    ).coalesce()

def sparse_eye(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Constructs an n x n identity matrix as a sparse PyTorch COO tensor."""
    indices = torch.arange(n, device=device).repeat(2, 1)
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=(n, n), dtype=dtype, device=device
    ).coalesce()

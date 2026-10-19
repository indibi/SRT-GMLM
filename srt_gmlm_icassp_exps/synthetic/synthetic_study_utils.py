# pylint: disable=invalid-name, import-error
"""Resolvers used to parse experiment and study configurations."""
import math
from typing import Any

from omegaconf import OmegaConf
import torch
import networkx as nx
import optuna

from t_regs.manifolds import Steifel, Euclidean, GeneralizedSteifel
# from t_regs.solvers.manifold.rada_v3 import RADA_RGD
from t_regs.solvers.manifold.rada import RADA_RGD
from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
# from t_regs.models.regression.tucker_nn_bcd_v2 import TuckerRegressorBCD
from t_regs.models.regression.tucker_nn_bcd import TuckerRegressorBCD
from t_regs.synthetic_data.generate_lr_data import (
    generate_sparse_low_tucker_rank_tensor
    )
from t_regs.synthetic_data.regression import (
    generate_tensor_regression_predictors,
    generate_tensor_regression_response
    )


def get_factor_nx_graph(_parent_) -> nx.Graph:
    """Create a graph for each mode according to the configuration"""
    if _parent_.type == 'grid':
        G = nx.grid_graph(_parent_.dim,
                          periodic= _parent_.get('periodic', False)
                          )
    elif _parent_.type == 'grid_2d':
        G = nx.grid_2d_graph(*(_parent_.dim),
                             periodic= _parent_.get('periodic', False)
                             )
    else:
        raise NotImplementedError(
            "Only grid and grid_2d options are implemented for yaml configuration."
            )
    return G

def get_laplacian_matrix(
        G: nx.Graph | nx.DiGraph,
        dtype_str: str = 'double'
        ) -> torch.Tensor:
    """Create laplacian matrix from networkx graph."""
    L = torch.tensor(
        nx.laplacian_matrix(G).toarray(),
        dtype=getattr(torch, dtype_str),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    return L

def get_gen_steifel_weight_matrix(L:torch.Tensor, alpha:float) -> torch.Tensor:
    """Create weight matrix W = (I + alpha L) for generalized steifel manifold"""
    I = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    W = I + alpha*L
    return W

def init_rada_rgd(rada_rgd_cfg: dict[str, Any],
                               lda: float,
                               dim: int) -> dict[str, Any]:
    rada_rgd_cfg['R'] = lda*(dim)**0.5 if lda!=0 else 1e-6*(dim)**0.5
    rada_rgd_cfg['beta1'] = 0.01*(dim)**0.5
    return RADA_RGD(**rada_rgd_cfg)#rada_rgd_cfg


def register_resolvers():
    """Register the resolvers used to parse the configuration files."""
    OmegaConf.clear_resolvers()
    # OmegaConf.register_new_resolver(
    #     'eval', eval
    # )
    OmegaConf.register_new_resolver(
        'len', len
    )
    OmegaConf.register_new_resolver(
        'mul', lambda *args: math.prod(args)
    )
    OmegaConf.register_new_resolver(
        'mul_list', lambda lst: math.prod(lst)
    )
    OmegaConf.register_new_resolver(
        'populate_list_of_len', lambda obj, length: [obj]*length
    )
    OmegaConf.register_new_resolver(
        'divide', lambda x,y: x/y
    )
    OmegaConf.register_new_resolver(
        'minus', lambda x,y: x - y
    )
    OmegaConf.register_new_resolver(
        'RADA_RGD', lambda kwargs: RADA_RGD(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'get_factor_nx_graph', get_factor_nx_graph
    )
    OmegaConf.register_new_resolver(
        'get_laplacian_matrix', get_laplacian_matrix
    )
    OmegaConf.register_new_resolver(
        'get_gen_steifel_weight_matrix', get_gen_steifel_weight_matrix
    )
    OmegaConf.register_new_resolver(
        'Steifel', lambda n, p, device: Steifel(n, p, device=device)
    )
    OmegaConf.register_new_resolver(
        'GeneralizedSteifel', lambda kwargs: GeneralizedSteifel(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'Euclidean', lambda kwargs: Euclidean(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'GridSampler', lambda kwargs : optuna.samplers.GridSampler(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'init_rada_rgd', lambda cfg, lda, dim: init_rada_rgd(
            cfg, lda, dim
        )
    )
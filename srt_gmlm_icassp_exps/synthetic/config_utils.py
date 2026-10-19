# pylint: disable=invalid-name, import-error
"""Resolvers used to parse experiment and study configurations."""
from omegaconf import OmegaConf
import torch
import networkx as nx
import numpy as np

# from t_regs.models.regression.tucker_bcd_gen_steifel import GenTuckerBCD
from t_regs.solvers.manifold import RiemmannianGradientDescent as RGD
from t_regs.solvers.manifold.line_searcher import LineSearcher, ArmijoPointSearch
from t_regs.manifolds import Steifel, Euclidean, GeneralizedSteifel
from t_regs.solvers.manifold.minimax_problem import MinimaxProblem
from t_regs.solvers.manifold.rada import RADA_RGD


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


def register_resolvers():
    """Register the resolvers used to parse the configuration files."""
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver(
        'eval', eval
    )
    OmegaConf.register_new_resolver(
        'len', len
    )
    OmegaConf.register_new_resolver(
        'logspace', lambda start, stop, num: list(np.logspace(start, stop, num))
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
        'RGD', lambda kwargs: RGD(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'RADA_RGD', lambda kwargs: RADA_RGD(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'LineSearcher', lambda kwargs: LineSearcher(**kwargs)
    )
    OmegaConf.register_new_resolver(
        'ArmijoPointSearch', lambda kwargs: ArmijoPointSearch(**kwargs)
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

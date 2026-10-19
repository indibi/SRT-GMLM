import os, sys
import argparse
from collections import defaultdict
from typing import Any, Dict, Optional, Callable
from time import perf_counter
from copy import deepcopy
from itertools import product
from pprint import pprint
from pathlib import Path

import omegaconf as oc
from omegaconf import OmegaConf
import pandas as pd
import torch
import dask
import wandb
from dotenv import load_dotenv

from dask.distributed import Client, as_completed


from synthetic_study_utils import register_resolvers
from hp_study_utils import get_data, get_model

from t_regs.models.regression.tucker_nn_regressor import TuckerRegressor
from t_regs.models.regression.tucker_nn_bcd_v2 import TuckerRegressorBCD
from t_regs.models.regression.tucker.treg_nn_bcd_model_selection import TuckerBCDRegularizerSearch

def parse_and_load_exp_config():
    description=(
        "`TuckerRegressorBCD` with Regularizer Search Experiment on synthetic "
        "regression task. The experiment configuration is specified with a YAML "
        "file. Provide the yaml filename (with extension) with the command line "
        "argument. The config files can be overriden with dot-list style command"
        " line arguments."
        )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Filename of the YAML configuration file.')
    parser.add_argument(
        '--seeds',
        type=str,
        required=True,
        help=('Comma separated list of seeds to use for repeated experiment '
        'trials. E.g. "0,1,2" will run the experiment with seeds 0,1, and 2.'
        )
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help=(
            'If provided, experiment creates a new yaml config file with the '
            'provided name and populates it with the overriden settings.'
            )
        )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results if the study name already exists.')
    parser.add_argument(
        '--devices',
        type=str,
        default= None,
        help=('List of devices to use for the study. Provide as comma separated '
            'string, e.g. "cuda:0,cuda:1" or "cpu".')
        )
    

    args, unknown_args = parser.parse_known_args()
    if args.devices is None:
        if torch.cuda.device_count():
            devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            devices = ['cpu']
    else:
        devices = [d.strip() for d in args.devices.split(',')] 
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    # Load configuration
    script_path = Path(__file__).parent
    cfg_path = script_path / 'configs'/ args.config
    results_dir = script_path / 'results'
    if not cfg_path.is_file():
        print(f"Configuration file {args.config} does not exist.")
        sys.exit(1)
        if not results_dir.is_dir():
            print(f"Results directory {results_dir} does not exist. Creating it.")
            results_dir.mkdir(parents=True, exist_ok=True)
    
    # Register omegaconf resolvers before loading the configuration
    register_resolvers()

    config = OmegaConf.load(cfg_path)
    if args.exp_name is not None:
        config.hdr.name = args.exp_name
    
    if len(unknown_args) > 0:
        print("Overriding study configuration with command line arguments:")
        for arg in unknown_args:
            print(arg)

    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))

    if len(unknown_args) > 0:
        print("Creating a new configuration file with the overridden settings.")
        file_name = f"{config.hdr.group}-{config.hdr.name}.yaml"
        OmegaConf.save(config,
                       script_path / 'configs' / file_name,
                       resolve=False
                       )
    print("Experiment Configuration:")
    print("-"*80)
    print("")
    print(OmegaConf.to_yaml(config))
    print("")
    print("-"*80)
    print("")
    return config, seeds, results_dir, devices


def run_exp_for_ivs_and_model(config, wandb_kwargs, control_vals,model_key,device):
    register_resolvers()
    model_cfg = config.models[model_key]
    data_cfg = config.data

    data = get_data(data_cfg)
    data['X_train'] = data['X'][:data_cfg.n_train].to(device)
    data['y_train'] = data['y'][:data_cfg.n_train].to(device).reshape(-1, 1)

    if model_cfg.model_name == 'TuckerBCDRegularizerSearch':
        regsearch_args = model_cfg.regsearch_args
        regsearch = TuckerBCDRegularizerSearch(**regsearch_args)
        base_model_cfg = model_cfg.base_model_cfg
        base_model_cfg.regressor.feature_ranks = model_cfg.feature_ranks
        regsearch.search(
            get_model(model_cfg.base_model_cfg, device),
            X=data['X_train'].to(device),
            y=data['y_train'].to(device),
            X_test=data['X_test'].to(device) if data['X_test'] is not None else None,
            y_test=data['y_test'].to(device).reshape(-1, 1) if data['y_test'] is not None else None,
            control_vals=control_vals,
            wandb_kwargs=wandb_kwargs,
        )
        df: pd.DataFrame = regsearch.summary_df

    return df, device


def set_config_values_from_ivs(config, values, var_names, cfg_keys):
    conf = deepcopy(config)
    for var_name, var_val in zip(var_names, values):
        if var_name not in cfg_keys:
            raise ValueError(f"Variable name {var_name} not found in cfg_keys.")
        keys = cfg_keys[var_name]
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            OmegaConf.update(conf, key, var_val, merge=False)
    return conf


def main():
    exp_config, seeds,  results_dir, devices = parse_and_load_exp_config()
    ivs = exp_config.ivs
    cvs = exp_config.cvs
    variables = ivs.variables
    exp_name = exp_config.hdr.name
    client = Client(n_workers=5, threads_per_worker=2, memory_limit='8GB')  # start local workers as processes

    available_devs = set(devices)
    jobs_in_queue = []
    for seed in seeds:
        exp_cfg = deepcopy(exp_config)
        exp_cfg.cvs.seed = seed

        iv_names = ivs.variables
        # iv_cfg_keys = [ivs.cfg_keys[name] for name in iv_names]
        iv_ranges = [ivs.ranges[name] for name in iv_names]

        control_vars = {'seed':seed}

        for i, vals_tuple in enumerate(product(*iv_ranges)):
            for j, iv_name in enumerate(iv_names):
                for cfg_key in exp_cfg.ivs.cfg_keys[iv_name]:
                    OmegaConf.update(exp_cfg, cfg_key, vals_tuple[j])
                control_vars[iv_name] = vals_tuple[j]
            for k, model_key in enumerate(exp_cfg.models.keys()):
                exp_cfg.data.seed = seed    

                wandb_kwargs = {
                    'project': exp_config.hdr.project,
                    'group': exp_config.hdr.group,
                    'name': f"{exp_config.hdr.name}-s{seed}-ivs{i}",
                    'config': {**control_vars, **exp_cfg.cvs}
                }

                jobs_in_queue.append([run_exp_for_ivs_and_model,
                                    deepcopy(exp_cfg),
                                    deepcopy(wandb_kwargs),
                                    deepcopy(control_vars),
                                    model_key
                    ])
    futures = []
    for i in range(min(len(devices), len(jobs_in_queue))):
        futures.append(
            client.submit(
                *(jobs_in_queue.pop(0)), devices[i]
            )
        )
        available_devs.remove(devices[i])
    ac = as_completed(futures)

    for future in ac:
        df, device = future.result()
        if df is not None:
            results_file = results_dir / f"{exp_config.hdr.group}-{exp_config.hdr.name}.csv"
            df.to_csv(results_file,
                mode='a',
                index=False,
                header =not os.path.exists(results_file)
                )
        
        if len(jobs_in_queue) >0:
            ac.add(client.submit(*jobs_in_queue.pop(0), device))


if __name__ == "__main__":
    register_resolvers()
    load_dotenv()
    os.environ['WANDB_INIT_TIMEOUT'] = '600'
    wandb.login()
    main()
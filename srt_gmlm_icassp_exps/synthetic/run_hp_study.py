import os, sys
import argparse
from collections import defaultdict
from typing import Any, Dict, Optional, Callable
from time import perf_counter
from copy import deepcopy
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import omegaconf as oc
from omegaconf import OmegaConf


from synthetic_study_utils import register_resolvers
from hp_study_utils import HyperparameterStudy

def main():
    description=(
        "Hyperparameter study on synthetic tensor regression experiments. "
        "The study configuration must be provided as a YAML file. Provide the "
        "filename (with extension) as a command line argument. Config files can "
        "be overriden with dot-list style command line arguments. "
        )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Filename of the YAML configuration file.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results if the study name already exists.')
    parser.add_argument(
        '--devices',
        type=str,
        default='cuda:0,cuda:1,cuda:2,cuda:3',
        help=('List of devices to use for the study. Provide as comma separated '
            'string, e.g. "cuda:0,cuda:1" or "cpu".')
        )

    args, unknown_args = parser.parse_known_args()
    devices = [d.strip() for d in args.devices.split(',')]
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
    config.study_header.results_dir = str(results_dir)
    if len(unknown_args) > 0:
        print("Overriding study configuration with command line arguments:")
        for arg in unknown_args:
            print(arg)
    print("Study Configuration:")
    print("-"*80)
    print("")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))
    print(OmegaConf.to_yaml(config))
    print("")
    print("-"*80)

    # update study name based on unknown args
    if len(unknown_args) > 0:
        postfix = '_'.join(
            [arg.replace('--', '').replace('=', '_') for arg in unknown_args]
            )
        study_name = config.study_header.study_name
        config.study_header.study_name = f"{study_name}_{postfix}"
    # # Initialize and run the hyperparameter study
    hp_study = HyperparameterStudy(config, devices, overwrite=args.overwrite)
    hp_study.run_study()

if __name__ == "__main__":
    main()
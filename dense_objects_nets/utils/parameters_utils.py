# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import numpy as np

from copy import deepcopy
from typing import Dict, Any, List, Tuple


def set_nested_dict(dict, dict_keys, value):
    """Recursive function to set the value for the last key in a a sequence of nested dict keys."""
    next_key = dict_keys.pop(0) if dict_keys else None
            
    if next_key == None:
        return value
    
    dict[next_key] = set_nested_dict(dict[next_key], dict_keys, value)
    return dict

def combine_training_parameters(params_dict: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    """
    Given a dict of 'parameters entries' creates a list containing the needed information for all the possible different
    parameters values combinations. 
    
    See '$WS_DIR/config/training_runs/_training_run_example.yml' for a description of all the config parameters.

    Args:
    - params_dict (`dict`): dictionary containing all the parameters that we want to combine.

    Returns:
    - `list`: list containing the combined parameters information.
    """
    # Get all parameters values combinations
    combined_params = None
    params_names = []
    for key, data in params_dict.items():
        if "values" in data:
            p_values = data["values"]
            p_names = data["names"] if "names" in data else [f"{value}" for value in p_values]
        else:
            min_v, max_v, step, tag = float(data["min_value"]), float(data["max_value"]), float(data["step"]), str(data["tag"])
            adjusted_max = min_v + ((max_v - min_v) // step) * step
            p_values =  [f"{v:08.5f}" for v in np.arange(min_v,  adjusted_max + step/2, step)] # Allow max_v to be included
            p_names = [f"{tag}{value}" for value in p_values]

        # Fist param/iteration
        if combined_params == None:
            combined_params = []

            for v, n in zip(p_values, p_names):
                combined_params.append(([key], [v]))
                params_names.append(n)

            continue

        new_combinations = []
        new_names = []
        # Combine current param values with all previous ones
        for (keys, values), names in zip(combined_params, params_names):
            for v, n in zip(p_values, p_names):
                new_combinations.append((keys + [key], values + [v]))
                new_names.append(f"{names}_{n}")

        combined_params = new_combinations
        params_names = new_names
   
    return combined_params, params_names

def create_combined_training_configs(combined_parameters: Tuple[List[List[str]], List[str]], base_training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Creates the training config files from the given cross combined parameters. 

    Args:
    - combined_parameters (`list`): list containing the combined parameters information.
    - base_training_config (`dict`): training config file used as a base when overriding the combination of parameters values.  

    Returns:
    - `list`: list of produced training config files.
    """
    # Foreach parameters combination create the required training and dataset scenes config files.
    training_config_files = []
    for keys, values in combined_parameters:
        # Update the base training config file setting all the current combination of parameters values.
        training_config = deepcopy(base_training_config)
        for k, v in zip(keys, values):
            nested_keys = k.split("/")
            # Convert value to float if possible, otherwise use the original.
            try:
                value = float(v)
            except (ValueError, TypeError):
                value = v
            training_config = set_nested_dict(dict=training_config, dict_keys=nested_keys, value=value)
        
        # Save the created outputs files with the 
        training_config_files.append(training_config)
        
    return training_config_files

def override_config_parameter(config: Dict[str, Any], param_key: str, param_value: Any) -> Dict[str, Any]:
    """
    Overrides a parameter value in a nested config dictionary.

    Args:
    - config (`dict`): The original configuration dictionary.
    - param_key (`str`): The key of the parameter to override, using '/' to denote nested keys.
    - param_value (`Any`): The new value for the parameter.

    Returns:
    - `dict`: The updated configuration dictionary with the overridden parameter.
    """
    nested_keys = param_key.split("/")
    updated_config = set_nested_dict(dict=deepcopy(config), dict_keys=nested_keys, value=param_value)
    return updated_config
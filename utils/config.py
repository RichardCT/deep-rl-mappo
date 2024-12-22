import yaml
import os

import utils.enums as enums

def parse_config(config_file):
    """
    Parse the YAML configuration file.

    Args:
    config_file (str): Path to the YAML configuration file.

    Returns:
    dict: Parsed configuration.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Validate configuration
    validate_config(config)

    return config


def validate_config(config):
    """
    Validate the configuration.

    Args:
    config (dict): Configuration dictionary.

    Raises:
    ValueError: If a required field is missing or has an invalid type.
    """
    required_fields = {
        'env': ['id', 'type'],
        'training': ['num_envs', 'total_timesteps', 'num_rollout_steps', 'update_epochs', 'num_minibatches'],
        'optimization': ['learning_rate', 'gamma', 'gae_lambda', 'surrogate_clip_threshold', 
                         'entropy_loss_coefficient', 'value_function_loss_coefficient', 
                         'normalize_advantages', 'clip_value_function_loss', 'max_grad_norm', 
                         'target_kl', 'anneal_lr', 'rpo_alpha'],
        'simulation': ['seed', 'torch_deterministic', 'capture_video', 'use_tensorboard', 'save_model']
    }

    type_checks = {
        'env': {'id': str, 'type': int},
        'training': {'num_envs': int, 'total_timesteps': int, 'num_rollout_steps': int, 
                     'update_epochs': int, 'num_minibatches': int},
        'optimization': {'learning_rate': float, 'gamma': float, 'gae_lambda': float, 
                         'surrogate_clip_threshold': float, 'entropy_loss_coefficient': float, 
                         'value_function_loss_coefficient': float, 'normalize_advantages': bool, 
                         'clip_value_function_loss': bool, 'max_grad_norm': float, 
                         'target_kl': (float, type(None)), 'anneal_lr': bool, 'rpo_alpha': (float, type(None))},
        'simulation': {'seed': int, 'torch_deterministic': bool, 'capture_video': bool, 
                       'use_tensorboard': bool, 'save_model': bool}
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing section: {section}")

        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing field: {section}.{field}")

            field_type = type_checks[section][field]
            if not isinstance(config[section][field], field_type):
                raise ValueError(f"Invalid type for {section}.{field}: expected {field_type}, got {type(config[section][field])}")
            
    if config['env']['type'] == enums.EnvType.DISCRETE and config['optimization']['rpo_alpha'] is not None:
        print(
            f"rpo_alpha is not used in discrete environments. Ignoring rpo_alpha={config['optimization']['rpo_alpha']}"
        )

if __name__ == '__main__':
    config_file = 'hyp.yaml'
    config = parse_config(config_file)
    print(config)
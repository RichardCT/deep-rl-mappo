import os
import time

import numpy as np
import torch
import torch.optim as optim

from ppo import PPO
from mappo import MAPPO

import utils.helpers as helpers
import utils.env as env_utils
import utils.agent as agent_utils
import utils.eval as eval_utils
import utils.config as config_utils
import utils.enums as enums
import utils.plot as plot_utils

from classes.data_cache import DataCache

def setup():
    config = config_utils.parse_config("hyp.yaml")
    data_cache = DataCache()
    exp_name, run_name = helpers.get_run_name(config)
    if not os.path.exists(f"runs/{run_name}/{exp_name}"):
        os.makedirs(f"runs/{run_name}/{exp_name}")
    helpers.set_seed(config['simulation']['seed'], config['simulation']['torch_deterministic'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config, data_cache, exp_name, run_name, device

def main():
    """
    Main function to run the PPO (Proximal Policy Optimization) algorithm.

    This function sets up the environment, creates the PPO agent, and runs the training process.
    It handles both discrete and continuous action spaces, and includes options for
    various PPO algorithm parameters and training configurations.

    Args:
        # Environment parameters
        env_id (str): Identifier for the Gymnasium environment to use.
        env_is_discrete (bool): Whether the environment has a discrete action space.
        num_envs (int): Number of parallel environments to run.

        # Core training parameters
        total_timesteps (int): Total number of timesteps to run the training for. This is the number of interactions with the environment
        num_rollout_steps (int): Number of steps to run in each environment per rollout.
        update_epochs (int): Number of epochs to update the policy for each rollout.
        num_minibatches (int): Number of minibatches for each update.

        # Core PPO algorithm parameters
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        surrogate_clip_threshold (float): Clipping parameter for the surrogate objective.

        # Loss function coefficients
        entropy_loss_coefficient (float): Coefficient for the entropy term in the loss.
        value_function_loss_coefficient (float): Coefficient for the value function loss.

        # Advanced PPO parameters
        normalize_advantages (bool): Whether to normalize advantages.
        clip_value_function_loss (bool): Whether to clip the value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.
        target_kl (float): Target KL divergence for early stopping, if not None.

        # Learning rate schedule
        anneal_lr (bool): Whether to use learning rate annealing.

        # Continuous action space specific
        rpo_alpha (float): Alpha parameter for Regularized Policy Optimization (continuous action spaces only).

        # Reproducibility and logging
        seed (int): Random seed for reproducibility.
        torch_deterministic (bool): Whether to use deterministic algorithms in PyTorch.
        capture_video (bool): Whether to capture and save videos of the agent's performance.
        use_tensorboard (bool): Whether to use TensorBoard for logging.
        save_model (bool): Whether to save the trained model to disk and validate this by running a simple evaluation.
    """
    config, data_cache, exp_name, run_name, device = setup()
    print(device)
    train_envs = env_utils.create_envs(config, mode=enums.EnvMode.TRAIN, run_name=run_name)
    val_envs = env_utils.create_envs(config, mode=enums.EnvMode.VAL, run_name=run_name)

    # Assume you have a list of your environments

    agents = [
        agent_utils.get_agent(
            train_envs,
            config['env']['type'],
            config['optimization']['rpo_alpha'],
            device
        )
        for _ in range(config['env']['num_agents'])  # num_agents defines the number of agents
    ]

    # Initialize optimizers for all agents
    #optimizers = [
    #    optim.Adam(agent.parameters(), lr=config['optimization']['learning_rate'], eps=1e-5)
    #    for agent in agents
    #]

    mappo = MAPPO(
        agents=agents,  # Pass the list of agents
        optimizer=None,  # List of optimizers corresponding to each agent
        train_envs=train_envs,
        val_envs=val_envs,
        config=config,
        run_name=run_name,
        data_cache=data_cache
    )
    # Train the agent
    trained_agent = mappo.learn()
    # run_name = "FrozenLake-v1__v3__1__1730707240"
    # exp_name = "v3"
    model_path = f"runs/{run_name}/{exp_name}.rl_model"
    if config['simulation']['save_model']:
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    test_envs = env_utils.create_envs(config, mode=enums.EnvMode.TEST, run_name=run_name, data_cache=data_cache)
    
    eval_agent = agent_utils.get_agent(
        test_envs,
        config['env']['type'], 
        config['optimization']['rpo_alpha'], 
        device
    )
        
    eval_agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    eval_utils.evaluate_model(test_envs, eval_agent, config, data_cache, device)
    plot_utils.plot(run_name, data_cache)

if __name__ == "__main__": 
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils.enums as enums

from classes.linear_lr_schedule import LinearLRSchedule
from classes.ppo_logger import PPOLogger

class MAPPO:
    def __init__(
        self,
        agents,
        optimizer,
        train_envs,
        val_envs,
        config=None,
        run_name="run",
        data_cache=None,
    ):
    
        self.agents = agents
        self.num_agents = len(agents)
        self.train_envs = train_envs
        self.val_envs = val_envs
        #self.optimizer = optimizer
        self.seed = config['simulation']['seed']
        #self.centralized_critic = config['optimization']['centralized_critic']
        self.platform_position = None

        self.num_rollout_steps = config['training']['num_rollout_steps']
        self.num_envs = config['training']['num_envs']
        self.batch_size = self.num_envs * self.num_rollout_steps * self.num_agents
        self.num_minibatches = config['training']['num_minibatches']
        self.minibatch_size = self.batch_size // config['training']['num_minibatches']
        self.total_timesteps = config['training']['total_timesteps']

        self.gamma = config['optimization']['gamma']
        self.gae_lambda = config['optimization']['gae_lambda']
        self.surrogate_clip_threshold = config['optimization']['surrogate_clip_threshold']
        self.entropy_loss_coefficient = config['optimization']['entropy_loss_coefficient']
        self.value_function_loss_coefficient = config['optimization']['value_function_loss_coefficient']
        self.max_grad_norm = config['optimization']['max_grad_norm']
        self.update_epochs = config['training']['update_epochs']
        self.normalize_advantages = config['optimization']['normalize_advantages']
        self.clip_value_function_loss = config['optimization']['clip_value_function_loss']
        self.target_kl = config['optimization']['target_kl']

        self.device = next(agents[0].parameters()).device

        self.anneal_lr = config['optimization']['anneal_lr']
        self.initial_lr = config['optimization']['learning_rate']

        self.use_val = config['validation']['is_active']

        self.lr_scheduler = None
        self.logger = PPOLogger(run_name, config['simulation']['use_tensorboard'])

        self.global_step_t = 0
        
        self.num_policy_updates = self.total_timesteps // (self.num_rollout_steps * self.num_envs)
            
        self.data_cache = data_cache

        self.centralized_critic = nn.Sequential(
            nn.Linear(self.train_envs.single_observation_space.shape[0], 128),  # No need to multiply by self.num_agents
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.centralized_critic.parameters()) + [param for agent in self.agents for param in agent.parameters()],
            lr=self.initial_lr,
            eps=1e-5
        )

        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(self.num_policy_updates)

    def create_lr_scheduler(self, num_policy_updates):
        return LinearLRSchedule(self.optimizer, self.initial_lr, num_policy_updates)

    def learn(self):
        # Initialize environment observations for training and validation
        t_next_observation, t_is_next_observation_terminal = self._initialize_environment(self.train_envs)
        v_next_observation, v_is_next_observation_terminal = self._initialize_environment(self.val_envs)
        self.global_step_t = 0

        for _ in range(self.num_policy_updates):
            if self.anneal_lr:
                self.lr_scheduler.step()

            # Collect rollouts and update policy
            t_update_results = self.collect_rollouts_and_update_policy(
                t_next_observation, t_is_next_observation_terminal, self.train_envs, enums.EnvMode.TRAIN
            )

            if self.use_val:
                self.collect_rollouts(
                    v_combined_obs, v_is_next_observation_terminal, self.val_envs, enums.EnvMode.VAL
                )

            self.logger.log_policy_update(t_update_results, self.global_step_t)

        print(f"Training completed. Total steps: {self.global_step_t}")
        return self.agents


    def _initialize_environment(self, envs):
        initial_observation, _ = envs.reset(seed=self.seed)
        self.platform_position = self.train_envs.call("get_platform_position")
        # initial_observation shape: (num_envs, 786816)

        # Convert observations to tensors
        initial_observation = torch.Tensor(initial_observation).to(self.device)

        is_initial_observation_terminal = torch.zeros(self.num_envs).to(self.device)
        return initial_observation, is_initial_observation_terminal


    def collect_rollouts_and_update_policy(self, next_observation, is_next_observation_terminal, envs, mode):
        (flattened_data_per_agent, next_observation, is_next_observation_terminal) = self.collect_rollouts(next_observation, is_next_observation_terminal, envs, mode)
        update_results = self.update_policy(flattened_data_per_agent)
        return update_results

    def collect_rollouts(self, next_observations, is_next_observation_terminal, envs, mode):
        """
        Collect rollouts for multiple agents.
        """
        collected_data = {agent_id: self._initialize_storage(envs) for agent_id in range(self.num_agents)}

        for step in range(self.num_rollout_steps):
            actions = []
            for agent_id in range(self.num_agents):
                collected_data[agent_id]['observations'][step] = next_observations

                # Store the termination flag for the current agent
                collected_data[agent_id]['terminals'][step] = is_next_observation_terminal

                with torch.no_grad():
                    action, logprob = self.agents[agent_id].sample_action_and_compute_log_prob(next_observations)
                    value = self.agents[agent_id].estimate_value_from_observation(next_observations)

                # Flatten the value if needed
                collected_data[agent_id]['values'][step] = value.flatten()

                # Store action and log probability
                collected_data[agent_id]['actions'][step] = action
                collected_data[agent_id]['log_probabilities'][step] = logprob

                action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                actions.append(action)
            
            actions = tuple(actions)

                # Execute the environment step
            next_observation, reward, terminations, truncations, infos = envs.step(actions)
            next_observation = next_observation.reshape(self.num_envs, -1)

            if mode == enums.EnvMode.TRAIN:
                self.global_step_t += self.num_envs

            for agent_id in range(self.num_agents):
                # Update rewards based on platform logic
                agent_positions = self.train_envs.call("get_agent_position", agent_id)  # Add a method to get agent position
                
                agent_rewards = reward[:, agent_id]  # Shape (4,)

                for i, (pos) in enumerate(agent_positions):
                    if pos == self.platform_position:
                        agent_rewards[i] += 0.5  # Example platform reward

                # Update rewards and next terminal status
                collected_data[agent_id]['rewards'][step] = torch.tensor(agent_rewards, device=self.device)

            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(self.device)
            is_next_observation_terminal = torch.tensor(
                np.logical_or(terminations, truncations), dtype=torch.float32, device=self.device
            )

            #combined_observations = torch.cat([next_observations[agent_id] for agent_id in range(self.num_agents)], dim=-1)
            # Use centralized critic to estimate the value
            global_value_estimate = self.centralized_critic(next_observation)
            for agent_id in range(self.num_agents):
                collected_data[agent_id]['values'][step] = global_value_estimate.flatten()

            # Handle final information (outside the agent loop)
            self.logger.log_rollout_step(infos, self.global_step_t, mode)
            if "final_info" in infos:
                for env_index, env_final_info in enumerate(infos["final_info"]):
                    if isinstance(env_final_info, dict) and "episode" in env_final_info:
                        # Process episode rewards for each agent
                        for agent_id, reward in enumerate(env_final_info["episode"]["r"]):
                            self.data_cache.cache_reward(agent_id, reward, mode)

        # Compute advantages and returns after rollout collection
        with torch.no_grad():
            next_values = {}
            for agent_id in range(self.num_agents):
                # Estimate the value of the next state for each agent
                next_values[agent_id] = self.agents[agent_id].estimate_value_from_observation(
                    next_observation
                ).reshape(1, -1)

            # Compute advantages and returns for all agents
            advantages, returns = {}, {}
            for agent_id in range(self.num_agents):
                advantages[agent_id], returns[agent_id] = self.compute_advantages(
                    collected_data[agent_id]['rewards'],
                    collected_data[agent_id]['values'],
                    collected_data[agent_id]['terminals'],
                    next_values[agent_id],
                    is_next_observation_terminal,
                )

        flattened_data_per_agent  = self._flatten_multi_agent_rollout_data(collected_data, advantages, returns)
        return (flattened_data_per_agent, next_observation, is_next_observation_terminal,
        )
        #return self._flatten_multi_agent_rollout_data(collected_data, advantages, returns)

    def _initialize_storage(self, envs):
        """
        Adjust storage for multiple agents.
        """
        # Handle observation space dtype and shape
        if envs.single_observation_space.dtype == 'int64':
            obs_shape = np.array([envs.single_observation_space])
        else:
            obs_shape = envs.single_observation_space

        return {
            'observations': torch.zeros((self.num_rollout_steps, self.num_envs) + obs_shape.shape).to(self.device),
            #'actions': torch.zeros((self.num_rollout_steps, self.num_envs, envs.single_action_space[0].n)).to(self.device),
            'actions': torch.zeros((self.num_rollout_steps, self.num_envs), dtype=torch.long).to(self.device),
            'log_probabilities': torch.zeros(self.num_rollout_steps, self.num_envs).to(self.device),
            'rewards': torch.zeros(self.num_rollout_steps, self.num_envs).to(self.device),
            'terminals': torch.zeros(self.num_rollout_steps, self.num_envs).to(self.device),
            'values': torch.zeros(self.num_rollout_steps, self.num_envs).to(self.device),
        }

    def compute_advantages(self, rewards, values, terminals, next_value, is_terminal):
        """
        Compute advantages and returns using GAE for a single agent.

        Args:
            rewards (torch.Tensor): Rewards for the agent at each timestep. Shape: (num_rollout_steps, num_envs)
            values (torch.Tensor): Value estimates for the agent at each timestep. Shape: (num_rollout_steps, num_envs)
            terminals (torch.Tensor): Terminal flags for the agent at each timestep. Shape: (num_rollout_steps, num_envs)
            next_value (torch.Tensor): Value estimate for the next observation after the last step. Shape: (num_envs,)
            is_terminal (torch.Tensor): Terminal flag for the next observation. Shape: (num_envs,)

        Returns:
            tuple: Advantages and returns tensors for the agent. Shape: (num_rollout_steps, num_envs)
        """
        advantages = torch.zeros_like(rewards).to(self.device)  # Shape: (num_rollout_steps, num_envs)
        returns = torch.zeros_like(rewards).to(self.device)     # Shape: (num_rollout_steps, num_envs)

        gae = torch.zeros(rewards.shape[1]).to(self.device)  # Shape: (num_envs,)
        for t in reversed(range(self.num_rollout_steps)):
            if t == self.num_rollout_steps - 1:
                # For the last step, use the provided next_value and whether it is terminal
                episode_continues = 1.0 - is_terminal  # Shape: (num_envs,)
                next_values = next_value  # Shape: (num_envs,)
            else:
                # For all other steps, use values and terminal flags from the next step
                episode_continues = 1.0 - terminals[t + 1]  # Shape: (num_rollout_steps, num_envs)
                next_values = values[t + 1]  # Shape: (num_rollout_steps, num_envs)

            delta = rewards[t] + self.gamma * next_values * episode_continues - values[t]
            gae = delta + self.gamma * self.gae_lambda * episode_continues * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _flatten_multi_agent_rollout_data(self, collected_data, advantages, returns):
        """
        Flatten rollout data for all agents in a multi-agent setup.

        Args:
            collected_data (dict): Collected rollout data for all agents.
            advantages (dict): Computed advantages for all agents.
            returns (dict): Computed returns for all agents.

        Returns:
            dict: Flattened rollout data per agent.
        """
        flattened_data_per_agent = {}

        for agent_id in range(self.num_agents):
            agent_data = collected_data[agent_id]
            # Flatten observations and other data as needed
            flat_observations = agent_data['observations'].reshape(-1, agent_data['observations'].shape[-1])
            flat_log_probabilities = agent_data['log_probabilities'].reshape(-1)
            flat_actions = agent_data['actions'].reshape(-1)
            flat_advantages = advantages[agent_id].reshape(-1)
            flat_returns = returns[agent_id].reshape(-1)
            flat_values = agent_data['values'].reshape(-1)

            flattened_data_per_agent[agent_id] = {
                'observations': flat_observations,
                'log_probabilities': flat_log_probabilities,
                'actions': flat_actions,
                'advantages': flat_advantages,
                'returns': flat_returns,
                'values': flat_values,
            }

        return flattened_data_per_agent


    def update_policy(self, collected_data):
        shared_observations = collected_data[0]['observations']
        batch_size = shared_observations.shape[0]
        batch_indices = np.arange(batch_size)

        # Track metrics
        aggregated_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "old_approx_kl": [],
            "approx_kl": [],
            "clipping_fractions": [],
            "explained_variance": [],
        }

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                minibatch_observations = shared_observations[minibatch_indices]
                # Compute value estimates using the centralized critic
                new_values = self.centralized_critic(minibatch_observations).squeeze(-1)

                # Compute policy losses for each agent
                policy_losses, entropy_losses = [], []
                for agent_id in range(self.num_agents):
                    agent_observations = collected_data[agent_id]['observations'][minibatch_indices]
                    agent_actions = collected_data[agent_id]['actions'][minibatch_indices]
                    old_log_probs = collected_data[agent_id]['log_probabilities'][minibatch_indices]
                    advantages = collected_data[agent_id]['advantages'][minibatch_indices]

                    # Normalize advantages per agent if needed
                    if self.normalize_advantages:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Compute new log probs and entropy
                    current_policy_log_probs, entropy = self.agents[agent_id].compute_action_log_probabilities_and_entropy(
                        agent_observations,
                        agent_actions,
                    )

                    log_probability_ratio = current_policy_log_probs - old_log_probs
                    probability_ratio = torch.exp(log_probability_ratio)

                    # Compute policy loss
                    policy_loss = self.calculate_policy_gradient_loss(
                        advantages, probability_ratio
                    )
                    policy_losses.append(policy_loss)
                    entropy_losses.append(entropy.mean())

                    # Compute KL divergences
                    with torch.no_grad():
                        old_approx_kl = (-log_probability_ratio).mean()
                        approx_kl = ((probability_ratio - 1) - log_probability_ratio).mean()
                        clipping_fraction = ((probability_ratio - 1.0).abs() > self.surrogate_clip_threshold).float().mean().item()

                    # Track metrics
                    aggregated_metrics["old_approx_kl"].append(old_approx_kl.item())
                    aggregated_metrics["approx_kl"].append(approx_kl.item())
                    aggregated_metrics["clipping_fractions"].append(clipping_fraction)

                    # Early stopping based on KL divergence
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break

                # Aggregate losses
                policy_gradient_loss = sum(policy_losses) / len(policy_losses)
                entropy_loss = sum(entropy_losses) / len(entropy_losses)

                # Compute value function loss using centralized critic
                minibatch_returns = torch.cat(
                    [collected_data[agent_id]['returns'][minibatch_indices] for agent_id in range(self.num_agents)],
                    dim=0
                )
                value_function_loss = self.calculate_value_function_loss(
                    new_values, minibatch_returns, 
                    torch.cat([collected_data[agent_id]['values'][minibatch_indices] for agent_id in range(self.num_agents)]),
                    minibatch_indices
                )

                # Total loss
                total_loss = (
                    policy_gradient_loss
                    - self.entropy_loss_coefficient * entropy_loss
                    + self.value_function_loss_coefficient * value_function_loss
                )

                # Compute metrics before backward
                with torch.no_grad():
                    predicted_values = new_values.detach().cpu().numpy()
                    actual_returns = minibatch_returns.cpu().numpy()
                    observed_return_variance = np.var(actual_returns)

                    explained_variance = (
                        np.nan
                        if observed_return_variance == 0
                        else 1 - np.var(actual_returns - predicted_values) / observed_return_variance
                    )
                    aggregated_metrics["explained_variance"].append(explained_variance)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.centralized_critic.parameters()) + [param for agent in self.agents for param in agent.parameters()],
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Metrics for minibatch
                aggregated_metrics["policy_loss"].append(policy_gradient_loss.item())
                aggregated_metrics["value_loss"].append(value_function_loss.item())
                aggregated_metrics["entropy_loss"].append(entropy_loss.item())

        # Aggregate metrics across minibatches
        final_metrics = {key: np.mean(value) for key, value in aggregated_metrics.items()}
        return final_metrics



    def calculate_policy_gradient_loss(self, minibatch_advantages, probability_ratio):
        """
        Calculate the policy gradient loss using the PPO clipped objective, which is designed to
        improve the stability of policy updates. It uses a clipped surrogate objective
        that limits the incentive for the new policy to deviate too far from the old policy.

        Args:
            minibatch_advantages (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the advantage estimates for each sample in the minibatch.
            probability_ratio (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the ratio of probabilities under the new and old policies for each action.

        Returns:
            torch.Tensor: A scalar tensor containing the computed policy gradient loss.

        The PPO loss is defined as:
        L^CLIP(θ) = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        Where:
        - r_t(θ) is the probability ratio
        - A_t is the advantage estimate
        - ε is the surrogate_clip_threshold
        """

        # L^PG(θ) = r_t(θ) * A_t
        # This is the standard policy gradient objective. It encourages
        # the policy to increase the probability of actions that led to higher
        # advantages (i.e., performed better than expected).
        unclipped_pg_obj = minibatch_advantages * probability_ratio

        # L^CLIP(θ) = clip(r_t(θ), 1-ε, 1+ε) * A_t
        # This limits how much the policy can change for each action.
        # If an action's probability increased/decreased too much compared to
        # the old policy, we clip it. This prevents drastic policy changes,
        # promoting more stable learning.
        clipped_pg_obj = minibatch_advantages * torch.clamp(
            probability_ratio,
            1 - self.surrogate_clip_threshold,
            1 + self.surrogate_clip_threshold,
        )

        # L^CLIP(θ) = -min(L^PG(θ), L^CLIP(θ))
        # Use the minimum of the clipped and unclipped objectives.
        # By taking the minimum and then negating (for gradient ascent),
        # we choose the more pessimistic (lower) estimate.
        # This ensures that:
        # 1. We don't overly reward actions just because they had high advantages
        #    (unclipped loss might do this).
        # 2. We don't ignore actions where the policy changed a lot if they still
        #    result in a worse objective (clipped loss might do this).
        # This conservative approach helps prevent the policy from changing too
        # rapidly in any direction, improving stability.
        policy_gradient_loss = -torch.min(unclipped_pg_obj, clipped_pg_obj).mean()

        return policy_gradient_loss

    def calculate_value_function_loss(
        self, new_value, computed_returns, previous_value_estimates, minibatch_indices
    ):
        """
        Calculate the value function loss, optionally with clipping, for the value function approximation.
        It uses either a simple MSE loss or a clipped version similar to the policy loss clipping
        in PPO. When clipping is enabled, it uses the maximum of clipped and unclipped losses.
        The clipping helps to prevent the value function from changing too much in a single update.


        Args:
            new_value (torch.Tensor): Tensor of shape (minibatch_size,) containing
                the new value estimates for the sampled states.
            computed_returns (torch.Tensor): Tensor of shape (batch_size,) containing
                the computed returns for each step in the rollout.
            previous_value_estimates (torch.Tensor): Tensor of shape (batch_size,)
                containing the value estimates from the previous iteration.
            minibatch_indices (np.array): Array of indices for the current minibatch.

        Returns:
            torch.Tensor: A scalar tensor containing the computed value function loss.

        The value function loss is defined as:
        If clipping is enabled:
        L^VF = 0.5 * E[max((V_θ(s_t) - R_t)^2, (clip(V_θ(s_t) - V_old(s_t), -ε, ε) + V_old(s_t) - R_t)^2)]
        If clipping is disabled:
        L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]

        Where:
        - V_θ(s_t) is the new value estimate
        - R_t is the computed return
        - V_old(s_t) is the old value estimate
        - ε is the surrogate_clip_threshold
        """
        new_value = new_value.view(-1)

        if self.clip_value_function_loss:
            # L^VF_unclipped = (V_θ(s_t) - R_t)^2
            # This is the standard MSE loss, pushing the value estimate
            # towards the actual observed returns.
            unclipped_vf_loss = (new_value - computed_returns[minibatch_indices]) ** 2

            # V_clipped = V_old(s_t) + clip(V_θ(s_t) - V_old(s_t), -ε, ε)
            # This limits how much the value estimate can change from its
            # previous value, promoting stability in learning.
            clipped_value_diff = torch.clamp(
                new_value - previous_value_estimates[minibatch_indices],
                -self.surrogate_clip_threshold,
                self.surrogate_clip_threshold,
            )
            clipped_value = (
                previous_value_estimates[minibatch_indices] + clipped_value_diff
            )

            # L^VF_clipped = (V_clipped - R_t)^2
            # This loss encourages updates within the clipped range, preventing drastic changes to the value function.
            clipped_vf_loss = (clipped_value - computed_returns[minibatch_indices]) ** 2

            # L^VF = max(L^VF_unclipped, L^VF_clipped)
            # By taking the maximum, we choose the more pessimistic (larger) loss.
            # This ensures we don't ignore large errors outside the clipped range
            # while still benefiting from clipping's stability.
            v_loss_max = torch.max(unclipped_vf_loss, clipped_vf_loss)

            # The 0.5 factor simplifies the gradient of the squared error loss,
            # as it cancels out with the 2 from the derivative of x^2.
            value_function_loss = 0.5 * v_loss_max.mean()
        else:
            # If not clipping, use simple MSE loss
            # L^VF = 0.5 * E[(V_θ(s_t) - R_t)^2]
            # Intuition: Without clipping, we directly encourage the value function
            # to predict the observed returns as accurately as possible.
            value_function_loss = (
                0.5 * ((new_value - computed_returns[minibatch_indices]) ** 2).mean()
            )

        return value_function_loss
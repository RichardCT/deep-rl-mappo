from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils.helpers import preprocess_observation


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module, ABC):
    @abstractmethod
    def estimate_value_from_observation(self, observation):
        """
        Estimate the value of an observation using the critic network.

        Args:
            observation: The observation to estimate.

        Returns:
            The estimated value of the observation.
        """
        pass

    @abstractmethod
    def get_action_distribution(self, observation):
        """
        Get the action distribution for a given observation.

        Args:
            observation: The observation to base the action distribution on.

        Returns:
            A probability distribution over possible actions.
        """
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations):
        """
        Sample an action from the action distribution and compute its log probability.

        Args:
            observations: The observations to base the actions on.

        Returns:
            A tuple containing:
            - The sampled action(s)
            - The log probability of the sampled action(s)
        """
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        """
        Compute the log probabilities and entropy of given actions for given observations.

        Args:
            observations: The observations corresponding to the actions.
            actions: The actions to compute probabilities and entropy for.

        Returns:
            A tuple containing:
            - The log probabilities of the actions
            - The entropy of the action distribution
        """
        pass


class DiscreteAgent(BaseAgent):
    def __init__(self, envs):
        super().__init__()
        if envs.single_observation_space.dtype == 'int64':
            a = np.array([envs.single_observation_space])
        else:
            a = envs.single_observation_space
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(a.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(a.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def estimate_value_from_observation(self, observation):
        return self.critic(observation)

    def get_action_distribution(self, observation):
        logits = self.actor(observation)
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy

class ImgAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),  # Output: (32, 30, 30)
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # Output: (64, 14, 14)
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # Output: (64, 12, 12)
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.img_size = envs.envs[0].img_size

    def estimate_value_from_observation(self, x):
        x = preprocess_observation(x, self.img_size)
        return self.critic(self.network(x))
    
    def get_action_distribution(self, x):
        x = preprocess_observation(x, self.img_size)
        logits = self.actor(self.network(x))
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy
    
class LLMv2Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # Define CNN network for the image part after resizing to (128, 128)
        self.image_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4)),  # Output: (32, 30, 30)
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),  # Output: (64, 14, 14)
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),  # Output: (64, 12, 12)
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 12 * 12, 512)),
            nn.ReLU(),
        )

        # Define linear network for the vector input
        self.linear_net = nn.Sequential(
            layer_init(nn.Linear(384, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        # Combine CNN and linear outputs
        self.combined_fc = nn.Sequential(
            layer_init(nn.Linear(512 + 128, 512)),  # 512 from CNN, 128 from linear net
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.single_action_space[0].n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.img_size = envs.envs[0].img_size

    def estimate_value_from_observation(self, x):
        return self.critic(self.combine(x))
    
    def get_action_distribution(self, x):
        logits = self.actor(self.combine(x))
        return Categorical(logits=logits)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, entropy

    def combine(self, x):
        image_input = x[:, :512*512*3].view(-1, 3, 512, 512)
        vector_input = x[:, 512*512*3:]

        image_input = preprocess_observation(image_input, self.img_size)
        image_output = self.image_cnn(image_input)
        
        vector_output = self.linear_net(vector_input)

        combined_input = torch.cat((image_output, vector_output), dim=1)
        combined_output = self.combined_fc(combined_input)
        
        return combined_output

class ContinuousAgent(BaseAgent):
    def __init__(self, envs, rpo_alpha=None):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def estimate_value_from_observation(self, observation):
        return self.critic(observation)

    def get_action_distribution(self, observation):
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_dist = Normal(action_mean, action_std)

        return action_dist

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)

        if self.rpo_alpha is not None:
            # sample again to add stochasticity to the policy
            action_mean = action_dist.mean
            z = (
                torch.FloatTensor(action_mean.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(self.actor_logstd.device)
            )
            action_mean = action_mean + z
            action_dist = Normal(action_mean, action_dist.stddev)

        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy

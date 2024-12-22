import utils.enums as enums
import numpy as np

class DataCache:
    def __init__(self):
        self.train_rewards = {0: [], 1: []}  # Separate rewards for each agent
        self.val_rewards = {0: [], 1: []}
        self.test_rewards = {0: [], 1: []}
        
        self.agent_policy = {0: {}, 1: {}}  # Separate policy tracking for each agent
    
    def cache_reward(self, agent_id, reward, mode):
        if isinstance(reward, np.ndarray):
            reward = reward.ravel()
        if mode == enums.EnvMode.TRAIN:
            self.train_rewards[agent_id].extend(reward)
        elif mode == enums.EnvMode.VAL:
            self.val_rewards[agent_id].extend(reward)
        else:
            self.test_rewards[agent_id].extend(reward)
    
    def cache_policy(self, agent_id, x, y, action):
        if (x, y) not in self.agent_policy[agent_id]:
            self.agent_policy[agent_id][(x, y)] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.agent_policy[agent_id][(x, y)][action] += 1
import os

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))

from utils.helpers import load_npy_files_to_dict
import utils.enums as enums
import utils.render_env as render_utils
import utils.env as env_utils

class ImgEnv(gym.Env):
    def __init__(self, env_id, run_name='runs', mode=enums.EnvMode.TRAIN, 
                 use_pre_computed_states=True, size = 4, is_random = False,
                 seed=42, data_cache=None, is_slippery=False):
        super().__init__()
        self.env_id = env_id
        self.run_name = run_name
        self.size = size
        self.is_random = is_random
        self.current_map = ["SFFF", "FHFH", "FFFH", "HFFG"]
        self.current_map_id = "SFFFFHFHFFFHHFFG"
        self.mode = mode
        self.episodes = 0
        self.is_slippery = is_slippery

        self.img_size = size*64
        self.img_resolution = 128
        self.render_mode='rgb_array'
        
        self.seed = seed
        random.seed(seed)
        
        self.env = None
        self.env = self.init_env(env_id)
        
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3*self.img_size**2,), dtype=np.float32)
        self.action_space = self.env.action_space
        
        self.states = {}
        self.current_action_str = "0_1"
        
        self.state_path = f"states/img_env/{env_utils.mode_str_from_enum(self.mode)}"
        if not os._exists(f"{self.state_path}"):
            os.makedirs(f"{self.state_path}", exist_ok=True)
        
        if use_pre_computed_states:
            self.load_precomputed_states()
            
        self.data_cache = data_cache
                    
    def init_env(self, env_id):
        if self.is_random:
            self.randomize_map()
        env = gym.make(env_id, render_mode="rgb_array", desc=self.current_map, is_slippery=self.is_slippery)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.unwrapped.__init__(render_mode="rgb_array", is_slippery=self.is_slippery, desc=self.current_map)
        return env
        
    def load_precomputed_states(self):
        if not os.path.exists(f"{self.state_path}"):
            print("Folder for precomputed states not found")
            return
        self.states = load_npy_files_to_dict(f"{self.state_path}/")
        
    def randomize_map(self):
        self.current_map = generate_random_map(size=self.size, p=0.7, seed=random.randint(0, 1000))
        self.current_map_id = "".join(self.current_map)
        
        if self.env is not None:
            self.env.unwrapped.__init__(render_mode="rgb_array", is_slippery=self.is_slippery, desc=self.current_map)      
        
    def reset(self, **kwargs):
        if self.is_random:
            self.randomize_map()
        observation, info = self.env.reset(**kwargs)
        self.current_action_str = f"{observation}_1"
        state = self.get_state()
        state = state.flatten()
        self.episodes += 1
        return state, info
    
    def get_state(self):
        key = f"{self.current_map_id}_{self.current_action_str}"
        if key not in self.states:
            img = render_utils.render_arr(self.current_map_id, self.current_action_str, self.mode)
            self.states[key] = img
            np.save(f"{self.state_path}/{key}", img)
        return self.states[key]        
    
    def step(self, action):
        next_observation, reward, termination, truncations, infos = self.env.step(action)
        self.current_action_str = f"{next_observation}_{action}"
        state = self.get_state()
        state = state.flatten()
        
        if self.data_cache is not None:
            x, y = int(next_observation % self.size), int(next_observation / self.size)
            self.data_cache.cache_policy(x, y, action)

        return state, reward, termination, truncations, infos
    
    def render(self):
        return self.get_state()

    def close(self):
        self.env.close()
        
if __name__ == '__main__':
    env = ImgEnv('FrozenLake-v1', 0, is_random=True)
    env.reset()
    # print(env.states['0_0'].shape)
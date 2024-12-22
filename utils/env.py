import gymnasium as gym
import numpy as np
import os

from classes.envs.llm_env import LLMEnv
from classes.envs.img_env import ImgEnv
from classes.envs.llmv2_env import LLMv2Env
import utils.helpers as helpers
import utils.enums as enums

def make_discrete_env(env_id, idx, capture_video, run_name):
    if capture_video and idx == 0:
        if env_id == 'FrozenLake-v1':
            env = gym.make(env_id, render_mode="rgb_array", is_slippery=False)
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        if env_id == 'FrozenLake-v1':
            env = gym.make(env_id, is_slippery=False)
        else:
            env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def make_continuous_env(env_id, idx, capture_video, run_name, gamma):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(
        env
    ) 
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def make_continuous_envs(env_id, capture_video, run_name, num_envs, gamma):
    envs = gym.vector.SyncVectorEnv(
        [
            make_continuous_env(env_id, i, capture_video, run_name, gamma)
            for i in range(num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

def make_discrete_envs(env_id, capture_video, run_name, num_envs):
    envs = gym.vector.SyncVectorEnv(
        lambda: make_discrete_env(env_id, i, capture_video, run_name) for i in range(num_envs)
    )
    assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"
    return envs

def make_llmv2_envs(env_id, capture_video, run_name, num_envs, use_pre_computed_states, size, is_random, is_ma, seed, mode=enums.EnvMode.TRAIN, data_cache=None, is_slippery=False):
    def make_llmv2_env(idx):
        env = LLMv2Env(env_id, run_name=run_name, use_pre_computed_states=use_pre_computed_states, size=size, is_random=is_random, is_ma=is_ma, mode=mode, seed=seed, data_cache=data_cache, is_slippery=is_slippery)
        if idx==0 and capture_video:
            video_path = f"videos/{run_name}/{mode_str_from_enum(mode)}"
            if not os._exists(video_path):
                os.makedirs(video_path, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_path)
        return env
    envs = gym.vector.SyncVectorEnv(
        lambda: make_llmv2_env(i) for i in range(num_envs)
    )
    return envs

def make_llm_envs(env_id, capture_video, run_name, num_envs, use_pre_computed_states, size, is_random, seed, mode=enums.EnvMode.TRAIN, data_cache=None, is_slippery=False):
    def make_llm_env(idx):
        env = LLMEnv(env_id, run_name=run_name, use_pre_computed_states=use_pre_computed_states, size=size, is_random=is_random, mode=mode, seed=seed, data_cache=data_cache, is_slippery=is_slippery)
        if idx==0 and capture_video:
            video_path = f"videos/{run_name}/{mode_str_from_enum(mode)}"
            if not os._exists(video_path):
                os.makedirs(video_path, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_path)
        return env
    envs = gym.vector.SyncVectorEnv(
        lambda: make_llm_env(i) for i in range(num_envs)
    )
    return envs

def make_img_envs(env_id, capture_video, run_name, num_envs, use_pre_computed_states, size, is_random, is_slippery, seed, mode=enums.EnvMode.TRAIN, data_cache=None):
    def make_img_env(idx):
        env = ImgEnv(env_id, run_name=run_name, use_pre_computed_states=use_pre_computed_states, 
                               size=size, is_random=is_random, mode=mode, seed=seed, data_cache=data_cache, is_slippery=is_slippery)
        if idx==0 and capture_video:
            video_path = f"videos/{run_name}/{mode_str_from_enum(mode)}"
            if not os._exists(video_path):
                os.makedirs(video_path, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_path)
        return env
    envs = gym.vector.SyncVectorEnv(
        lambda: make_img_env(i) for i in range(num_envs)
    )
    return envs

def create_envs(config, mode=enums.EnvMode.TRAIN, run_name='runs', data_cache=None):
    mode_key = mode_str_from_enum(mode)
    env_id = config['env']['id']
    seed = config['env']['seed']
    capture_video = config['simulation']['capture_video']
    num_envs = config[mode_key]['num_envs']
    env_type = config['env']['type']
    is_slippery = config['env']['is_slippery']
    if env_type == enums.EnvType.LLMv2.value:
        return make_llmv2_envs(env_id, capture_video, run_name, num_envs, seed=seed, use_pre_computed_states = config['simulation']['use_pre_computed_states'],
                              size = config['env']['size'], is_random=config['env']['is_random'], is_ma=config['env']['is_ma'], mode=mode, data_cache=data_cache, is_slippery=is_slippery)
    elif env_type == enums.EnvType.LLM.value:
        return make_llm_envs(env_id, capture_video, run_name, num_envs, seed=seed, use_pre_computed_states = config['simulation']['use_pre_computed_states'],
                              size = config['env']['size'], is_random=config['env']['is_random'], mode=mode, data_cache=data_cache, is_slippery=is_slippery)
    elif env_type  == enums.EnvType.DISCRETE.value:
        return make_discrete_envs(env_id, capture_video, run_name, num_envs)
    elif env_type  == enums.EnvType.CONTINUOUS.value:
        return make_continuous_envs(env_id, capture_video, run_name, num_envs, config['optimization']['gamma'])
    elif env_type  == enums.EnvType.IMG.value:
        return make_img_envs(env_id, capture_video, run_name, num_envs, seed=seed, use_pre_computed_states = config['simulation']['use_pre_computed_states'],
                              size = config['env']['size'], is_random=config['env']['is_random'], mode=mode, data_cache=data_cache, is_slippery=is_slippery)
    
def mode_str_from_enum(mode):
    if mode == enums.EnvMode.TRAIN:
        return 'training'
    elif mode == enums.EnvMode.TEST:
        return 'testing'
    elif mode == enums.EnvMode.VAL:
        return 'validation'
    
# def mode_enum_to_file_path(mode):
#     if mode == enums.EnvMode.TRAIN.value:
#         return 'light'
#     elif mode == enums.EnvMode.TEST.value:
#         return 'dark'
#     else:
#         raise ValueError(f"Invalid mode: {mode}")
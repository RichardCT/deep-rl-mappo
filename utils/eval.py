import numpy as np
import torch

import utils.env as env_utils
import utils.enums as enums

def get_NormalizeObservation_wrapper(self, env_num=0):
    return self.gym_sync_vec_env.envs[env_num].env.env.env


def get_obs_norm_rms_obj(self, env_num=0):
    return self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms


def set_obs_norm_rms_obj(self, rms_obj, env_num=0):
    self.get_NormalizeObservation_wrapper(env_num=env_num).obs_rms = rms_obj

def evaluate_model(envs, model, config, data_cache, device):
    model.eval()
    num_envs = config['testing']['num_envs']
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).reshape(num_envs, -1).to(device)
    
    episodic_returns = []
    num_episodes = config['testing']['num_episodes']
    while len(episodic_returns) < num_episodes:
        actions, _ = model.sample_action_and_compute_log_prob(
            torch.Tensor(obs).to(device)
        )
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        obs = torch.Tensor(obs).reshape(num_envs, -1).to(device)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"Eval episode {len(episodic_returns)}, episodic return: {info['episode']['r']}"
                    )
                    episodic_returns.append(info["episode"]["r"])
                    data_cache.cache_reward(info["episode"]["r"], mode=enums.EnvMode.TEST)

    envs.close()

def load_and_evaluate_model(
    run_name,
    env_id,
    env_is_discrete,
    envs,
    num_envs,
    agent_class,
    device,
    model_path,
    gamma,
    capture_video,
):
    # Run simple evaluation to demonstrate how to load and use a trained model
    eval_episodes = 10
    eval_envs = env_utils.create_envs(
        env_id=env_id,
        num_envs=1,
        env_is_discrete=env_is_discrete,
        capture_video=capture_video,
        run_name=f"{run_name}-eval",
        gamma=gamma,
    )

    if not env_is_discrete:
        # Update normalization stats for continuous environments
        avg_rms_obj = (
            np.mean([envs.get_obs_norm_rms_obj(i) for i in range(num_envs)]) / num_envs
        )
        eval_envs.set_obs_norm_rms_obj(avg_rms_obj)

    eval_agent = agent_class(eval_envs).to(device)
    eval_agent.load_state_dict(torch.load(model_path, map_location=device))
    eval_agent.eval()

    obs, _ = eval_envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _ = eval_agent.sample_action_and_compute_log_prob(
            torch.Tensor(obs).to(device)
        )
        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"Eval episode {len(episodic_returns)}, episodic return: {info['episode']['r']}"
                    )
                    episodic_returns.append(info["episode"]["r"])

    eval_envs.close()
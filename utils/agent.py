from agent import ContinuousAgent, DiscreteAgent, ImgAgent, LLMv2Agent
from functools import partial

import utils.enums as enums

def get_agent_class(env_type, rpo_alpha):
    switch = {
        enums.EnvType.DISCRETE.value: DiscreteAgent,
        enums.EnvType.CONTINUOUS.value: partial(ContinuousAgent, rpo_alpha=rpo_alpha),
        enums.EnvType.LLM.value: DiscreteAgent,
        enums.EnvType.IMG.value: ImgAgent,
        enums.EnvType.LLMv2.value: LLMv2Agent,
    }
    return switch[env_type]
    
def get_agent(envs, env_type, rpo_alpha, device):
    agent_class = get_agent_class(env_type, rpo_alpha)
    agent = agent_class(envs).to(device)
    return agent
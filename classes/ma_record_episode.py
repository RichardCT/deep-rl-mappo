from gymnasium.wrappers import RecordEpisodeStatistics

class MultiAgentRecordEpisodeStatistics(RecordEpisodeStatistics):
    def __init__(self, env, num_agents, deque_size=100):
        super().__init__(env, deque_size=deque_size)
        self.num_agents = num_agents
        self.episode_returns = [0.0 for _ in range(num_agents)]
        self.episode_lengths = [0 for _ in range(num_agents)]
    
        print("self.num_agents: ", self.num_agents)
        print("self.episode_returns: ", self.episode_returns)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        print("self.num_agents: ", self.num_agents)
        print("self.episode_returns: ", self.episode_returns)

        for agent_id in range(self.num_agents):
            self.episode_returns[agent_id] += rewards[agent_id]
            self.episode_lengths[agent_id] += 1

            if terminations[agent_id] or truncations:
                self.return_queue.append(self.episode_returns[agent_id])
                self.length_queue.append(self.episode_lengths[agent_id])
                self.episode_returns[agent_id] = 0.0
                self.episode_lengths[agent_id] = 0

        return observations, rewards, terminations, truncations, infos
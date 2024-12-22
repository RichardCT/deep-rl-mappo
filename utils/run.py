import numpy as np
from classes.converter import EnvironmentConverter

def run_loop(env, agent, n_games, N):
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_games):
        observation = env.reset()[0]
        if type(observation) == int:
            observation = np.array([observation])
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            if type(observation) == int:
                observation = np.array([observation])
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            
        print(f"episode {i}: score {score} | avg_score: {avg_score} | time_steps: {n_steps} | learning_steps: {learn_iters}")
        
        indices = [i + 1 for i in range(len(score_history))]
    return indices, score_history

def run_ppo_llm_loop(env, agent, n_games, N):
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    converter = EnvironmentConverter()
    env.reset()
    
    for i in range(n_games):
        done = False
        score = 0
        while not done:
            render = env.render()
            observation = converter.convert(render)
            action, prob, val = agent.choose_action(observation)
            _, reward, done, _, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            
        print(f"episode {i}: score {score} | avg_score: {avg_score} | time_steps: {n_steps} | learning_steps: {learn_iters}")
        
        indices = [i + 1 for i in range(len(score_history))]
    return indices, score_history
 
import matplotlib.pyplot as plt
import numpy as np

import os

def plot(run, data_cache):
    plot_rewards(data_cache.train_rewards, data_cache.val_rewards, data_cache.test_rewards, run)
    plot_policy(data_cache.agent_policy, run)
    plot_value(data_cache.agent_policy, run)
    
    
def get_smoothed_series(series, weight=0.95):
    if len(series) < 1:
        return []
    smoothed_series = []
    last = series[0]
    for v in series:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed_series.append(smoothed_val)
        last = smoothed_val
    return smoothed_series

def pad_list(data, target_length):
    if len(data) < target_length:
        padding = [data[0] for _ in range(target_length - len(data))]
        data = np.append(data, padding)
    return data

def get_rolling_avg(series, window):
    ret = np.cumsum(series, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window
    
def plot_rewards(train_rewards, val_rewards, test_rewards, run):
  plt.clf()
  
  weight = 0.95
  smoothed_train = get_smoothed_series(train_rewards, weight)
  smoothed_val = get_smoothed_series(val_rewards, weight)
  smoothed_test = get_smoothed_series(test_rewards, weight)
  
  window=25
  rolling_avg_train = get_rolling_avg(train_rewards, window)
  rolling_avg_val = get_rolling_avg(val_rewards, window)
  rolling_avg_test = get_rolling_avg(test_rewards, window)
  
  max_length_train = max(len(smoothed_train), len(rolling_avg_train), len(train_rewards))
  max_length_val = max(len(smoothed_val), len(rolling_avg_val), len(val_rewards))
  max_length_test = max(len(smoothed_test), len(rolling_avg_test), len(test_rewards))
  smoothed_train = pad_list(smoothed_train, max_length_train)
  smoothed_val = pad_list(smoothed_val, max_length_val)
  smoothed_test = pad_list(smoothed_test, max_length_test)
  rolling_avg_train = pad_list(rolling_avg_train, max_length_train)
  rolling_avg_val = pad_list(rolling_avg_val, max_length_val)
  rolling_avg_test = pad_list(rolling_avg_test, max_length_test)
  # Sample data and plot setup
  x1 = np.arange(1, max_length_train + 1)
  x2 = np.arange(1, max_length_val + 1)
  x3 = np.arange(1, max_length_test + 1)
  color = '#add8e6'

  # Create a figure with 2 subplots
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), dpi=150)

  # Plot train rewards on subplot 1
  ax1.plot(x1, train_rewards, label="True Train Rewards", color=color, linewidth=0.5)
  ax1.plot(x1, smoothed_train, label="Smoothed Rewards (w=.95)", color='#457b9d', linewidth=1.0)
#   ax1.plot(x1, rolling_avg_train, label="Rolling Average", color='#000000', linewidth=1.0)
  ax2.plot(x2, val_rewards, label="True Val Rewards", color=color, linewidth=0.5)
  ax2.plot(x2, smoothed_val, label="Smoothed Val Rewards", color='#457b9d', linewidth=1.0)
  ax3.plot(x3, test_rewards, label="True Test Rewards", color=color, linewidth=0.5)
  ax3.plot(x3, smoothed_test, label="Smoothed Test Rewards", color='#457b9d', linewidth=1.0)
#   ax2.plot(x2, rolling_avg_test, label="Rolling Average", color='#000000', linewidth=1.0)
    
  # Labels and Legend for subplot 1
  ax1.set_title("Train - Agent Reward During Training")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Reward")
  ax1.legend()
  ax1.grid(False)

  # Plot test rewards on subplot 2 (optional)

  # Labels and Legend for subplot 2
  ax2.set_title("Val - Agent Performance")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Reward")
  ax2.legend()
  ax2.grid(False)

  # Labels and Legend for subplot 3
  ax3.set_title("Test - Agent Performance")
  ax3.set_xlabel("Epoch")
  ax3.set_ylabel("Reward")
  ax3.legend()
  ax3.grid(False)

  # Save the plot
  if not os._exists(f"plots/{run}/"):
    os.makedirs(f"plots/{run}/", exist_ok=True)
  plt.tight_layout()  # Adjust spacing between subplots
  plt.savefig(f"plots/{run}/rewards.png", format="png")
  plt.close()
  
def plot_policy(agent_policy, run_folder='runs'):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    max_x = max(x for x, _ in agent_policy.keys())
    for (y, x), actions_dict in agent_policy.items():
        x += 1
        y = max_x - y + 1
        threshold = 1e-10
        total_value = sum(actions_dict.values()) + threshold
        X = np.array([x, x, x, x])
        Y = np.array([y, y, y, y])
        u = np.array([-1.0 * actions_dict[0], 0.0, 1.0 * actions_dict[2], 0])
        v = np.array([0.0, 1.0 * actions_dict[3], 0.0, -1.0 * actions_dict[1]])
        u /= total_value
        v /= total_value
        ax.quiver(X, Y, u, v, scale_units='xy', scale=2.0, width=0.003,
                      color='#457b9d', headwidth=4.0, headaxislength=2.0)

    ax.set_aspect('equal')
    ax.set_xlim(0, max_x + 2)
    ax.set_ylim(0, max(y for _, y in agent_policy.keys()) + 2)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(f"plots/{run_folder}/policy.png", format="png")
    
def plot_value(agent_policy, run_folder='runs'):
    x_coords = [x for x, y in agent_policy.keys()]
    y_coords = [y for x, y in agent_policy.keys()]
    max_x, max_y = max(x_coords), max(y_coords)
    grid_size = (max_x + 1, max_y + 1)

    # Initialize a grid with zeroes
    exploration_grid = np.zeros(grid_size)

    # Calculate total actions per position
    for (x, y), actions in agent_policy.items():
        exploration_grid[x, y] = sum(actions.values())
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(exploration_grid, cmap='Blues', origin='upper')
    plt.colorbar(label='Frequency of Visits')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.title('Exploration Heatmap')
    
    plt.savefig(f"plots/{run_folder}/exploration.png", format="png")
    
# 0 -> left
# 1 -> down
# 2 -> right
# 3 -> up

if __name__ == '__main__':
    test_policy = {
        (0, 0): {
            1: 15.0,
            2: 0.0,
            3: 0.0,
            0: 0.0
        }, 
        (0, 1): {
            1: 7.0,
            2: 0.0,
            3: 0.0,
            0: 0.0
        }, 
        (1, 0): {
            1: 5.0,
            2: 5.0,
            3: 2.0,
            0: 0.0
        },
        (1, 1): {
            1: 5.0,
            2: 0.0,
            3: 0.0,
            0: 0.0
        }
    }
    # plot_policy(test_policy, 'demo_run')
    plot_value(test_policy, 'demo_run')
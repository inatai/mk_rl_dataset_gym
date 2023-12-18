import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
from env.eco_CartPole import eco_CartPoleEnv
from env.eco_move_CartPole import eco_move_CartPoleEnv

import math
import time
import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from decision_transformer.models.test_DQN import test_DQN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


env_name = "eco_move_CartPole"
device = "cuda"



options = {}
if env_name == "CartPole-v1":
    env = gym.make(env_name)
    options = {
        'max_episode': 1000,
        'epsilon_a': 25,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.1, # ε-greedyに使用
        'eps_end': 0.9,
        'eps_decay': 1000,
        'tau': 0.005, # ターゲットネットワークに使用
        'lr': 1e-4,
        'done_score': None, # このスコアに到達したら終了 Noneならmax_episodeまで学習する
        # 'done_scores': [100, 200, 300, 400, 500]
        'done_scores': None
    }
elif env_name == "s_CartPole":
    env = stationary_CartPoleEnv()
    options = {
        'max_episode': 10000,
        'epsilon_a': 25,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.1,
        'eps_end': 0.9,
        'tau': 0.005,
        'lr': 1e-4,
        'done_score': None,
        'done_scores': [300, 600, 900, 1200, 1500]
    }
elif env_name == "move_CartPole":
    env = move_CartPoleEnv()
    options = {
        'max_episode': 10000,
        'epsilon_a': 25,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.1,
        'eps_end': 0.9,
        'tau': 0.005,
        'lr': 1e-4,
        'done_score': None,
        'done_scores': [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]
    }
elif env_name == "eco_CartPole":
    env = eco_CartPoleEnv()
    options = {
        'max_episode': 10000,
        'epsilon_a': 25,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.1,
        'eps_end': 0.9,
        'tau': 0.005,
        'lr': 1e-4,
        'done_score': None,
        'done_scores': [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    }
elif env_name == "eco_move_CartPole":
    env = eco_move_CartPoleEnv()
    options = {
        'max_episode': 10000,
        'epsilon_a': 25,
        'batch': 128,
        'gamma': 0.9,
        'eps_start': 0.1,
        'eps_end': 0.9,
        'tau': 0.005,
        'lr': 1e-4,
        'done_score': None,
        'done_scores': [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    }
else:
    assert "envを選択してください"

# ステップ数の範囲を生成
steps_done_values = np.arange(0, options['max_episode'], 1)
eps_value = options['eps_end'] + (options['eps_start'] - options['eps_end']) * np.exp(-1. * steps_done_values / (options['max_episode'] * options['epsilon_a'] / 100))

save_folder = f'data/weight/{env_name}/DQN'
if not os.path.exists(save_folder): os.makedirs(save_folder)
# 最後に保存されるpath
if options['done_score']:
    save_model_name = f'{env_name}-{device}-score{options["done_score"]}-DQN.pth'
elif options['done_scores']:
    save_model_name = f'{env_name}-{device}-done_maxiter-DQN.pth'
else:
    save_model_name = f'{env_name}-{device}-{options["max_episode"]}epo-DQN.pth'
save_model_path = f'{save_folder}/{save_model_name}'






def main():
    start_time = time.time()

    done_scores_counter = 0
    for i_episode in range(options['max_episode']):
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        reward_sum = 0
        for t in count():
            action, epsilon = select_action(state, i_episode)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward_sum += reward
            
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*options['tau'] + target_net_state_dict[key]*(1-options['tau'])
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if i_episode%10 == 0:
                    seconds = int(time.time() - start_time)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    print(f'epi{i_episode} : [reward:{reward_sum},  episode_len:{t+1}, epsilon:{round(epsilon, 3)}], elapsed time:{hours:02}:{minutes:02}:{seconds:02}')
                break
        


        # この終了のさせ方だと、スコアだけ出して学習しないまま終了している可能性がある
        if options['done_score'] and reward_sum >= options['done_score']:
            break
        elif options['done_scores'] and done_scores_counter < len(options['done_scores']) and reward_sum >= options['done_scores'][done_scores_counter]:
            print(reward_sum)
            print(options["done_scores"][done_scores_counter])
            done_scores_save_name = f'{env_name}-{device}-done{options["done_scores"][done_scores_counter]}-DQN.pth'
            done_scores_save_path = f'{save_folder}/{done_scores_save_name}'
            torch.save(target_net, done_scores_save_path)
            done_scores_counter += 1
        

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    torch.save(target_net, save_model_path)




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = test_DQN(n_observations, n_actions).to(device)
target_net = test_DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=options['lr'], amsgrad=True)
memory = ReplayMemory(10000)



#ε-greedy法に従って行動を選択(エピソードが進むごとに探索の割合を低く)
def select_action(state, i_episode):
    sample = random.random()

    eps_threshold = eps_value[i_episode]
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), eps_threshold



episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < options['batch']:
        return
    transitions = memory.sample(options['batch'])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(options['batch'], device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * options['gamma']) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()





if __name__ == "__main__":
    main()
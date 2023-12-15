#目的はへたくそのデータから上手なデータまで、幅広いデータを収集することなので、
#学習結果の重みなどは保存しない


import gymnasium as gym
import math
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from decision_transformer.models.test_DQN import test_DQN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


env_name = "CartPole-v1"
device = "cuda"



options = {}
if env_name == "CartPole-v1":
    env = gym.make(env_name)
    options = {
        'num_episode': 100,
        'batch': 128,
        'gamma': 0.99,
        'eps_start': 0.9, #ε-greedyに使用
        'eps_end': 0.05,
        'eps_decay': 1000,
        'tau': 0.005, #ターゲットNに使用
        'lr': 1e-4
    }

data_save_folder = f'data/pkl/train/{env_name}/DQN'
if not os.path.exists(data_save_folder): os.makedirs(data_save_folder)
data_save_model_name = f'train_data_{env_name}-{device}-{options["num_episode"]}epo-DQN.pkl'
data_save_model_path = f'{data_save_folder}/{data_save_model_name}'
os.makedirs(data_save_folder, exist_ok=True)


'''
データの形
[
    { # 1エピソード目
        'state': [[], [], [], ..., []],
        'action': [[], [], [], ..., []],
        'reward': [[], [], [], ..., []],
        'done': [[], [], [], ..., []],
    },
    { # 2エピソード目
        'state': [[], [], [], ..., []],
        'action': [[], [], [], ..., []],
        'reward': [[], [], [], ..., []],
        'done': [[], [], [], ..., []],
    },

]
'''



def main():


    data = []
    for i_episode in range(options['num_episode']):
        
        states, actions, rewards, dones = [], [], [], []

        state, _ = env.reset()
        for t in count():
            tensor_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            tensor_action = select_action(tensor_state)
            action = tensor_action.item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            tensor_reward = torch.tensor([reward], device=device)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            if terminated:
                next_state = None
            else:
                next_state = state
                tensor_next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(tensor_state, tensor_action, tensor_next_state, tensor_reward)

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
                plot_durations()
                break
        
        epi_data = {}
        epi_data['states'] = np.array(states)
        epi_data['actions'] = np.array(actions)
        epi_data['rewards'] = np.array(rewards)
        epi_data['dones'] = np.array(dones)

        data.append(epi_data)

        

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    with open(data_save_model_path, 'wb') as tf:
        pickle.dump(data, tf)




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


steps_done = 0


#ε-greedy法に従って行動を選択(エピソードが進むごとに探索の割合を低く)
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = options['eps_end'] + (options['eps_start'] - options['eps_end']) * \
        math.exp(-1. * steps_done / options['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)



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
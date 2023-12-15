#学習済みのモデルを使用してdecision学習用のデータを作成する


import gymnasium as gym
import torch
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def main():
    env_name = 'CartPole-v1'
    model_names = [
        'CartPole-v1-cuda-100epo-DQN.pth',
    ]
    mode = 'normal'
    num_mode = 1 # my_modeのとき使用
    N = 100

    for model_name in model_names:
        if mode == 'my_mode':
            output_folder = f"data/pkl/decision/{env_name}/my_mode"
            data, rewards_ave = my_mk_pkl(env_name, model_name, N, num_mode)
            output_file_name = f"my_mode-{model_name.replace('.pth', '')}-{N}epi-{rewards_ave}point.pkl"
        elif mode == 'normal':
            output_folder = f"data/pkl/decision/{env_name}/normal"
            data, rewards_ave = mk_pkl(env_name, model_name, N)
            output_file_name = f"{model_name.replace('.pth', '')}-{N}epi-{rewards_ave}point.pkl"
        else:
            assert "モードを選択"

        output_file_path = os.path.join(output_folder, output_file_name)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_file_path, 'wb') as tf:
            pickle.dump(data, tf)



def mk_pkl(env_name, model_name, N):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    model = torch.load(f'data/weight/{env_name}/DQN/{model_name}')
    model.eval()

    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        assert "Discreteじゃないので設定する"
    sum_reward_list = []
    data = []
    for _ in range(N):
        observations, next_observations, actions, rewards, terminals= [], [], [], [], []
        
        state, _ = env.reset()
        done = False
        sum_reward = 0


        while not done:
            observations.append(state)

            tensor_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(tensor_state).argmax().item()

            onehot_action = np.zeros(act_dim)
            onehot_action[action] = 1

            actions.append(onehot_action)

            observation, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = observation
            
            actions.append(onehot_action)
            rewards.append(reward)
            terminals.append(done)
            next_observations.append(next_state)

            state = next_state
            sum_reward += reward


        epi_data = {}
        epi_data['observations'] = np.array(observations)
        epi_data['next_observations'] = np.array(next_observations)
        epi_data['actions'] = np.array(actions)
        epi_data['rewards'] = np.array(rewards)
        epi_data['terminals'] = np.array(terminals)

        data.append(epi_data)

        sum_reward_list.append(sum_reward)
    env.close()
    rewards_average = np.mean(sum_reward_list)
    
    return data, int(rewards_average)


def my_mk_pkl(env_name, model_name, N, num_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    model = torch.load(f'data/weight/{env_name}/DQN/{model_name}')
    model.eval()

    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        assert "Discreteじゃないので設定する"
    sum_reward_list = []
    data = []
    for _ in range(N):
        state, _ = env.reset()
        done = False
        reward = 0
        sum_reward = 0

        observations, next_observations, actions, rewards, terminals, modes = [], [], [], [], [], []
        while not done:
            tensor_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(tensor_state).argmax().item()

            onehot_action = np.zeros(act_dim)
            onehot_action[action] = 1

            observations.append(state)
            actions.append(onehot_action)
            rewards.append(reward)
            terminals.append(done)

            modes.append(np.array(num_mode)) # 仮

            observation, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = observation

            next_observations.append(next_state)

            state = next_state
            sum_reward += reward


        # for i, a in enumerate(actions):
        #     if a[0][0].dtype != torch.float32:
        #         print('{i}あああ')
        #     if a[0][1].dtype != torch.float32:
        #         print('{i}いいい')
        #     # print(a[0][0].dtype)
        #     # print(a[0][1].dtype)

        # # print(len(actions[1][0]))
        # # print(actions[1][0])

        epi_data = {}
        epi_data['observations'] = np.array(observations)
        epi_data['next_observations'] = np.array(next_observations)
        epi_data['actions'] = np.array(actions)
        epi_data['rewards'] = np.array(rewards)
        epi_data['terminals'] = np.array(terminals)
        epi_data['modes'] = np.array(modes)

        data.append(epi_data)

        sum_reward_list.append(sum_reward)
    env.close()
    rewards_average = np.mean(sum_reward_list)
    
    return data, int(rewards_average)




class test_DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(test_DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




if __name__ == "__main__":
    main()
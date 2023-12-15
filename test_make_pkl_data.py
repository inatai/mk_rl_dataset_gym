#学習済みのモデルを使用してdecision学習用のデータを作成する
#これはrewardの与え方を最後にひとまとまりにして与える
#理由はDT_testのときにrtgを与えやすくするため


import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
import torch
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def main():
    env_name = 'move_CartPole'
    # models_dict = {
    #     300: 's_CartPole-cuda-done300-DQN.pth',
    #     600: 's_CartPole-cuda-done600-DQN.pth',
    #     900: 's_CartPole-cuda-done900-DQN.pth',
    #     1200: 's_CartPole-cuda-done1200-DQN.pth',
    #     1500: 's_CartPole-cuda-done1500-DQN.pth',
    # }
    models_dict = {
        300: 'move_CartPole-cuda-done300-DQN.pth',
        600: 'move_CartPole-cuda-done300-DQN.pth',
        900: 'move_CartPole-cuda-done600-DQN.pth',
        1200: 'move_CartPole-cuda-done900-DQN.pth',
        1300: 'move_CartPole-cuda-done2100-DQN.pth',
    }
    mode = 'my_mode'
    num_mode = 2 # my_modeのとき使用
    N = 9000
    reward_threshold = 100 # データの選別の際に、± r_tなら保存

    for model_tuple in models_dict.items():
        if mode == 'my_mode':
            output_folder = f"data/pkl/decision/{env_name}/my_mode"
            data, rewards_ave = my_mk_pkl(env_name, model_tuple, N, num_mode, reward_threshold)
            output_file_name = f"my_mode-{model_tuple[1].replace('.pth', '')}-{N}epi-{rewards_ave}point.pkl"
        elif mode == 'normal':
            output_folder = f"data/pkl/decision/{env_name}/normal"
            data, rewards_ave = mk_pkl(env_name, model_tuple, N, reward_threshold)
            output_file_name = f"{model_tuple[1].replace('.pth', '')}-{N}epi-{rewards_ave}point.pkl"
        else:
            assert "モードを選択"

        output_file_path = os.path.join(output_folder, output_file_name)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_file_path, 'wb') as tf:
            pickle.dump(data, tf)



def mk_pkl(env_name, model_tuple, N, reward_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if env_name == "CartPole-v1":
        env = gym.make(env_name)
    elif env_name == "s_CartPole":
        env = stationary_CartPoleEnv()
    elif env_name == "move_CartPole":
        env = move_CartPoleEnv()
    else:
        assert "envを選択してください"


    model = torch.load(f'data/weight/{env_name}/DQN/{model_tuple[1]}')
    model.eval()

    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        assert "Discreteじゃないので設定する"
    sum_reward_list = []
    data = []
    while(N > len(data)):
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

            observation, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated

            next_state = observation
            
            actions.append(onehot_action)
            rewards.append(0)
            terminals.append(done)
            next_observations.append(next_state)

            state = next_state
            sum_reward += reward


        # rewardの閾値でデータを保存するかどうか決める
        if model_tuple[0] - reward_threshold < sum_reward < model_tuple[0] + reward_threshold:
            rewards[-1] = sum_reward

            epi_data = {}
            epi_data['observations'] = np.array(observations)
            epi_data['next_observations'] = np.array(next_observations)
            epi_data['actions'] = np.array(actions)
            epi_data['rewards'] = np.array(rewards)
            epi_data['terminals'] = np.array(terminals)

            data.append(epi_data)

            sum_reward_list.append(sum_reward)
        else:
            continue


    env.close()
    rewards_average = np.mean(sum_reward_list)
    
    return data, int(rewards_average)



def my_mk_pkl(env_name, model_tuple, N, num_mode, reward_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name == "CartPole-v1":
        env = gym.make(env_name)
    elif env_name == "s_CartPole":
        env = stationary_CartPoleEnv()
    elif env_name == "move_CartPole":
        env = move_CartPoleEnv()
    else:
        assert "envを選択してください"

    model = torch.load(f'data/weight/{env_name}/DQN/{model_tuple[1]}')
    model.eval()

    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        assert "Discreteじゃないので設定する"

    sum_reward_list = []
    data = []
    while(N > len(data)):
        observations, next_observations, actions, rewards, terminals, modes = [], [], [], [], [], []
        
        state, _ = env.reset()
        done = False
        sum_reward = 0

        while not done:
            modes.append(np.array(num_mode)) # 仮
            observations.append(state)

            tensor_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(tensor_state).argmax().item()

            onehot_action = np.zeros(act_dim)
            onehot_action[action] = 1

            observation, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated

            next_state = observation
            
            actions.append(onehot_action)
            rewards.append(0)
            terminals.append(done)
            next_observations.append(next_state)

            state = next_state
            sum_reward += reward

        # rewardの閾値でデータを保存するかどうか決める

        if model_tuple[0] - reward_threshold < sum_reward < model_tuple[0] + reward_threshold:
            rewards[-1] = sum_reward
            epi_data = {}
            epi_data['observations'] = np.array(observations)
            epi_data['next_observations'] = np.array(next_observations)
            epi_data['actions'] = np.array(actions)
            epi_data['rewards'] = np.array(rewards)
            epi_data['terminals'] = np.array(terminals)
            epi_data['modes'] = np.array(modes)

            data.append(epi_data)

            sum_reward_list.append(sum_reward)

            if len(data) % 100 == 0:
                print(f'target score = {model_tuple[0]} : episode data num = {len(data)} : score = {sum_reward}')

        else:
            # print(f'exclusion score = {sum_reward}')
            continue

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
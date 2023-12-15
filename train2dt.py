# data/pkl/train/{env}/の中にあるデータをDT学習用に変換する
'''
[
    { # 1エピソード目
        'states': [[], [], [], ..., []],
        'actions': [[], [], [], ..., []],
        'rewards': [[], [], [], ..., []],
        'dones': [[], [], [], ..., []],
    },
    { # 2エピソード目
        'states': [[], [], [], ..., []],
        'actions': [[], [], [], ..., []],
        'rewards': [[], [], [], ..., []],
        'dones': [[], [], [], ..., []],
    },

]

上のデータを

下のデータに変換


[
    { # 1エピソード目
        'observations': [[], [], [], ..., []],
        'next_observations': [[], [], [], ..., []],
        'actions': [[], [], [], ..., []],
        'rewards': [[], [], [], ..., []],
        'terminals': [[], [], [], ..., []],
    },
    { # 2エピソード目
        'observations': [[], [], [], ..., []],
        'next_observations': [[], [], [], ..., []],
        'actions': [[], [], [], ..., []],
        'rewards': [[], [], [], ..., []],
        'terminals': [[], [], [], ..., []],
    },

]

'''



import pickle
import numpy as np
import os

env_name = 'CartPole-v1'

load_data_folder = f'data/pkl/train/{env_name}/DQN/case1-2'
load_data_name = 'train_data_CartPole-v1-cuda-100epo-DQN.pkl'
load_data_path = f'{load_data_folder}/{load_data_name}'

save_folder = f'data/pkl/decision/{env_name}/normal/case1-2'
if not os.path.exists(save_folder): os.makedirs(save_folder)
save_name = f'{load_data_name.replace(".pkl", "-2-dtData")}.pkl'
save_path = f'{save_folder}/{save_name}'


dt_data = []

with open(load_data_path, mode="rb") as f:
    train_data = pickle.load(f)
    action_space = len(np.unique(train_data[0]['actions']))

    for e in train_data:

        states = e['states']
        actions = e['actions']
        rewards = e['rewards']
        dones = e['dones']

        # stateをnext_stateに変換作業
        list_states = states.tolist()
        list_next_states = list_states[1:]
        list_next_states.append(None)
        next_states = np.array(list_next_states)

        # actionをone hot ベクトルに変換
        onehot_actions = np.eye(action_space)[actions]

        epi_data = {}
        epi_data['observations'] = np.array(states)
        epi_data['next_observations'] = np.array(next_states)
        epi_data['actions'] = np.array(onehot_actions)
        epi_data['rewards'] = np.array(rewards)
        epi_data['terminals'] = np.array(dones)

        dt_data.append(epi_data)
    
with open(save_path, 'wb') as tf:
    pickle.dump(dt_data, tf)


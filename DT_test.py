# 参考 https://torch.classcat.com/2022/11/08/huggingface-blog-decision-transformers/

import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
import torch
import os
import random
import numpy as np
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.my_decision_transformer import MyDecisionTransformer
from env.stationary_CartPole import stationary_CartPoleEnv
from statistics import mean

import random

import torch



def main():

    env_name = 'CartPole-v1'
    # env_name = 'move_CartPole'
    model_folder = f'data/weight/{env_name}/DT'
    model_name = 'case2-3-scale100.0-10iter-DT.pth'
    state_mean_std_folder = f'data/mean-std/{env_name}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = torch.load(f'{model_folder}/{model_name}')
    model.eval()

    model_type = "my_dt"
    is_render = False
    target_returns = [300, 600, 900, 1200]
    modes = [1, 2]
    use_envs = [stationary_CartPoleEnv(), move_CartPoleEnv()]
    N = 10 # それぞれのパラメータでの検証回数
    scale = 100.
    max_ep_len = 1000

    state_mean_path = f'{state_mean_std_folder}/{model_name.replace(".pth", "-mean.npy")}'
    state_std_path = f'{state_mean_std_folder}/{model_name.replace(".pth", "-std.npy")}'

    state_mean = np.load(state_mean_path)
    state_std = np.load(state_std_path)


    if model_type == "dt": modes = [0]
    mode_score_list = []
    for i, mode in enumerate(modes):
        env = use_envs[i]

        if isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape
            state_dim = state_dim[0]
        state_dim = env.observation_space.shape
        state_dim = state_dim[0]

        targets_score_list = []
        for rtg in target_returns:
            rtg_score_list = []
            for _ in range(N):
                if model_type == 'dt':
                    epi_return, epi_length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len = max_ep_len,
                        device = device,
                        target_return = rtg,
                        state_mean=state_mean,
                        state_std=state_std,
                        scale = scale,
                        is_render=is_render
                    )
                elif model_type == 'my_dt':
                    epi_return, epi_length = my_evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        mode,
                        model,
                        scale,
                        max_ep_len,
                        state_mean,
                        state_std,
                        device='cuda',
                        target_return=rtg,
                        is_render=is_render
                    )
                else:
                    raise "model_typeのミス"
                
                score = abs(rtg - epi_return)
                # print(f'target return = {rtg}, episode return = {epi_return}, score = {score}')  

                rtg_score_list.append(score)
            targets_score_list.append(rtg_score_list)
        mode_score_list.append(targets_score_list)

    cal_score(mode_score_list, modes, target_returns)




def cal_score(mode_score_list, modes, target_returns):
    print("-------------rtg score------------------")
    mode_scores = []
    for i, targets_list in enumerate(mode_score_list):
        mode = modes[i]
        rtg_scores = []
        for j, rtg_list in enumerate(targets_list):
            rtg = target_returns[j]
            r_s = mean(rtg_list)
            rtg_scores.append(r_s)
            print(f'mode = {mode}, rtg = {rtg} : score = {r_s}')
        m_s = mean(rtg_scores)
        mode_scores.append(m_s)
    
    print("------------mode score------------------")
    for i, mode_score in enumerate(mode_scores):
        mode = modes[i]
        print(f'mode = {mode} : score = {mode_score}')
    
    print("------------model score------------------")
    print(f'model score = {mean(mode_scores)}')
        




def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len,
        scale,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        is_render=False,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    # if mode == 'noise':
    #     state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return/scale
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        if is_render:
            env.render()

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )


        actions[-1] = action
        action = action.detach().cpu().numpy()
        
        step_action = action.argmax().item()
        state, reward, terminated, truncated, _  = env.step(step_action)
        done = terminated or truncated

        print(step_action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # if mode != 'delayed':
        #     pred_return = target_return[0,-1] - (reward/scale)
        # else:
        #     pred_return = target_return[0,-1] # 前回のrewardを抽出

        pred_return = target_return[0,-1]

        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)    # 前回のtrを追加してtarget_return更新
                                                                                        # [[1, 1, 1, 1]]:torch.Size([1, 4]) -> [1, 1, 1, 1, 1]:torch.Size([1, 5])
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length



def my_evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        mode,
        model,
        scale,
        max_ep_len=1000,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        is_render=False
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    mode = mode # 追加
    mode = torch.tensor(mode, dtype=torch.float32) # 追加


    state, _ = env.reset()
    # if mode == 'noise':
    #     state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    modes = torch.zeros(0, device=device, dtype=torch.float32) # 追加

    modes = torch.cat([modes, torch.zeros(1, device=device)])
    modes[-1] = mode # 追加

    ep_return = target_return/scale
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        if is_render:
            env.render()

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(# states, actions, rewards, modes, returns_to_go, timesteps
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            modes.to(dtype=torch.float32), # 追加
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()
        

        # state, reward, done, _ = env.step(action)
        step_action = action.argmax().item()
        state, reward, terminated, truncated, _  = env.step(step_action)
        done = terminated or truncated

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        modes = torch.cat([modes, torch.zeros(1, device=device)]) # 追加
        modes[-1] = mode # 追加
        rewards[-1] = reward

        # if mode != 'delayed':
        #     pred_return = target_return[0,-1] - (reward/scale)
        # else:
        pred_return = target_return[0,-1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length



if __name__ == "__main__":
    main()
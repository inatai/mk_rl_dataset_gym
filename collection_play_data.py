import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
from env.eco_CartPole import eco_CartPoleEnv
import keyboard
import time
import glob
import os
import re
import pickle
import numpy as np


env_name = "LunarLander-v2"
player_name = "inada"
play_time = 49

save_folder = f"data/play-data/{env_name}/{player_name}"
os.makedirs(save_folder, exist_ok=True)
save_file_pattern = f"{env_name}-{player_name}.pkl"




if env_name == "CartPole-v1":
    env = gym.make(env_name, render_mode="human")
    key_list = ["left", "right"]
elif env_name == "s_CartPole":
    env = stationary_CartPoleEnv(render_mode="human")
    key_list = ["left", "right"]
elif env_name == "move_CartPole":
    env = move_CartPoleEnv(render_mode="human")
    key_list = ["left", "right"]
elif env_name == "eco_CartPole":
    env = eco_CartPoleEnv(render_mode="human")
    key_list = ["left", "right", None] 
elif env_name == "LunarLander-v2":
    env = gym.make(env_name, render_mode="human")
    key_list = [None, "left", "up", "right"] 
else:   
    assert "envを選択してください"


def main():
    for key in key_list:
        if key == None:
            N_flag = True

    for _ in range(play_time):
        observation, _ = env.reset()
        done = False
        sum_reward = 0

        observations, next_observations, actions, rewards, terminals= [], [], [], [], []

        while not done:
            # 環境の描画
            env.render()

            observations.append(observation)

            if N_flag:
                action = N_get_action(key_list)
            else:
                action = get_action(key_list)

            onehot_action = np.zeros(len(key_list))
            onehot_action[action] = 1
            
            observation, reward, terminated, truncated, _  = env.step(action)

            done = terminated or truncated

            sum_reward += reward

            terminals.append(done)
            rewards.append(0)
            actions.append(onehot_action)
            next_observations.append(observation)

        rewards[-1] = sum_reward

        epi_data = {}
        epi_data['observations'] = np.array(observations)
        epi_data['next_observations'] = np.array(next_observations)
        epi_data['actions'] = np.array(actions)
        epi_data['rewards'] = np.array(rewards)
        epi_data['terminals'] = np.array(terminals)

        print(sum_reward)
        # env.close()

        new_file_name = make_filename_by_seq(save_folder, save_file_pattern)
        save_path = f"{save_folder}/{new_file_name}"
        with open(save_path, 'wb') as tf:
                pickle.dump(epi_data, tf)


def N_get_action(key_list):
    time.sleep(0.01)
    for i, key in enumerate(key_list):
        if key == None:
            action = i
            continue
        elif keyboard.is_pressed(key):
            action = i
            break
    return action
    
def get_action(key_list):
    key_pressed = False
    while(not key_pressed):
        for i, key in enumerate(key_list):
            if keyboard.is_pressed(key):
                action = i
                key_pressed = True
                break
    return action
    

def make_filename_by_seq(dirname, filename, seq_digit=5): #seq_digit=3で001,002, seq_digit=5で00001,00002
    filename_without_ext, ext = os.path.splitext(filename)
    
    pattern = f"{filename_without_ext}_([0-9]*){ext}"
    prog = re.compile(pattern)

    files = glob.glob(
        os.path.join(dirname, f"{filename_without_ext}_[0-9]*{ext}")
    )

    max_seq = -1
    for f in files:
        m = prog.match(os.path.basename(f))
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    
    new_filename = f"{filename_without_ext}_{max_seq+1:0{seq_digit}}{ext}"
    
    return new_filename


if __name__ == "__main__":
    main()
import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
from env.eco_CartPole import eco_CartPoleEnv

import torch
import random
from decision_transformer.models.test_DQN import test_DQN

import torch



def main():
    # env_name = 'CartPole-v1'
    # env_name = 's_CartPole'
    # env_name = 'move_CartPole'
    env_name = 'move_CartPole'
    save_path = 'move_CartPole-cuda-done1500-DQN.pth'
    test_save_model(env_name, save_path)

    # env = gym.make(env_name, render_mode="human")
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()  # render game screen
    #         action = env.action_space.sample()  # this is random action. replace here to your algorithm!
    #         observation, reward, terminated, truncated, _  = env.step(action)  # get reward and next scene
    #         if terminated or truncated:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break



def test_save_model(env_name, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name == 'CartPole-v1':
        env = gym.make(env_name, render_mode="human")
    elif env_name == 's_CartPole':
        env = stationary_CartPoleEnv(render_mode="human")
    elif env_name == 'move_CartPole':
        env = move_CartPoleEnv(render_mode="human")
    elif env_name == 'eco_CartPole':
        env = eco_CartPoleEnv(render_mode="human")

    
    model = torch.load(f'data/weight/{env_name}/DQN/{save_path}')
    model.eval()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    rewards = 0
    while not done:
        # 環境の描画
        env.render()

        action = model(state).argmax().item()
        observation, reward, terminated, truncated, _  = env.step(action)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state
        rewards += reward
        # print(reward)

    env.close()

    print(rewards)




if __name__ == "__main__":
    main()
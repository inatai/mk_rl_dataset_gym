import gymnasium as gym
from env.stationary_CartPole import stationary_CartPoleEnv
from env.move_CartPole import move_CartPoleEnv
from env.eco_CartPole import eco_CartPoleEnv
from env.eco_move_CartPole import eco_move_CartPoleEnv
import keyboard
import time


env_name = "LunarLander-v2"

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
elif env_name == "eco_move_CartPole":
    env = eco_move_CartPoleEnv(render_mode="human")
    key_list = ["left", "right", None] 
elif env_name == "LunarLander-v2":
    env = gym.make(env_name, render_mode="human")
    key_list = [None, "left", "up", "right"]
else:   
    assert "envを選択してください"


def main():

    _, _ = env.reset()

    for key in key_list:
        if key == None:
            N_flag = True

    done = False
    rewards = 0
    while not done:
        # 環境の描画
        env.render()

        if N_flag:
            action = N_get_action(key_list)
        else:
            action = get_action(key_list)
        
        _ , reward, terminated, truncated, _  = env.step(action)

        done = terminated or truncated

        print(reward)

        rewards += reward

        # print(reward)
    print(rewards)
    env.close()


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
    



if __name__ == "__main__":
    main()
import gym
from gym import spaces
import numpy as np

#データの評価に使うが、事前データを使用するのかしないのか
class TSPEnv(gym.Env):
    def __init__(self, num_cities):
        super(TSPEnv, self).__init__()

        self.INF = 1000000000
        self.num_cities = num_cities
        self.x_cities, self.y_cities = self._generate_city()

        self.start_city = 0
        self.visited = np.zeros(num_cities, dtype=bool)
        self.visited[self.start_city] = True
        self.route = [self.start_city]
        self.current = np.eye(self.num_cities)[self.start_city]

        self.sum_reward = 0
        
        self.action_space = spaces.Discrete(num_cities)
        shape = (num_cities*4,)
        self.observation_space = spaces.Box(low=0, high=self.INF, shape=shape, dtype=np.int32)
        self.reward_range = [-100, 100]
        # [現在位置のonehot, 巡回済みのonehot, x座標, y座標]

    def step(self, action):
        # print(type(action))
        x_action = int(action)
        # print(type(x_action))
        if self.visited[x_action]:
            return self._get_observation(), self._get_reward(-100), False, {}
        else:
            self.visited[x_action] = True
            self.route.append(x_action)
            self.current = np.eye(self.num_cities)[x_action]

            if all(self.visited):
                return self._get_observation(), self._get_reward(100), True, {}
            else:
                return self._get_observation(), self._get_reward(1), False, {}

    def reset(self):
        self.x_cities, self.y_cities = self._generate_city()
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.visited[self.start_city] = True
        self.route = [self.start_city]
        self.current = np.eye(self.num_cities)[self.start_city]
        self.sum_reward = 0
        return self._get_observation()

    def _get_observation(self):
        observation = np.concatenate([self.current, self.visited ,self.y_cities, self.x_cities])
        # observation = np.array(self.current, self.visited, self.y_cities, self.x_cities)
        return observation
    
    def _get_reward(self, score):
        reward = score
        self.sum_reward += reward
        return reward

    def _generate_city(self):
        coordinates = set()
        while len(coordinates) < self.num_cities:
            x = round(np.random.rand() * 10000)
            y = round(np.random.rand() * 10000)
            coordinates.add((x, y))
        x_array = np.array([coord[0] for coord in coordinates])
        y_array = np.array([coord[1] for coord in coordinates])
        return x_array, y_array
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def info(self):
        return {
            'route': self.route,
            'reward': self.sum_reward,
        }
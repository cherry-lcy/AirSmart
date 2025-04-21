import numpy as np
import gym
from collections import deque
import random

# Custom environment for air conditioner control
class ACControlEnv(gym.Env):
    def __init__(self, historical_temps, room_heat_capacity=1000, ac_capacity=10, \
                 outdoor_heat_coeff=0.5, target_temp=22, time_step=1.0, hist_len=10, ac_power = 1.0):
        super().__init__()

        # ensure the historical data is longer than the required length
        assert len(historical_temps) >= hist_len

        self.historical_temps = historical_temps
        self.room_heat_capacity = room_heat_capacity # unit: kJ/°C
        self.ac_capacity = ac_capacity # kW/°C
        self.outdoor_heat_coeff = outdoor_heat_coeff # unit: kW/°C
        self.target_temp = target_temp
        self.time_step = time_step # unit: day
        self.hist_len = hist_len
        # custom the temperature of air conditioner (16-30 degree celcius)
        self.action_space = gym.spaces.Discrete(15)
        # custom a space for air conditioner
        self.observation_space = gym.spaces.Box(
            low=np.array([[10.0, -10.0]]*hist_len, dtype=np.float32),
            high=np.array([[40.0, 40.0]]*hist_len, dtype=np.float32),
            shape=(hist_len,2),
            dtype=np.float32
        )
        # initialization
        self.current_step = 0
        self.indoor_temp = target_temp
        self.last_temp = target_temp
        self.history = deque(maxlen=hist_len) # store the data of past 10 days
        self.ac_power = ac_power # unit: kW

    def reset(self):
        # reset the state
        self.current_step = 0
        self.indoor_temp = self.historical_temps[self.current_step]+random.uniform(-5,5)
        self.indoor_temp = np.clip(self.indoor_temp, 10.0, 40.0)
        self.history.clear()
        for i in range(min(self.hist_len, len(self.historical_temps))):
            outdoor_temp = self.historical_temps[i]
            indoor_temp = outdoor_temp + random.uniform(-5,5)
            indoor_temp = np.clip(indoor_temp, 10.0, 40.0)
            self.history.append([indoor_temp, outdoor_temp])
        self.last_temp = self.indoor_temp
        return self._get_obs()

    def step(self, action):
        outdoor_temp = self.historical_temps[self.current_step]
        # Map action to temperature setting with noise
        noise = np.random.uniform(-1, 1)
        ac_temp = int(action) + 16 + noise
        ac_temp = np.clip(ac_temp, 16.0, 30.0)

        # Simulate indoor temperature change
        q_ac = self.ac_capacity * (ac_temp - self.indoor_temp) * self.time_step
        q_outdoor = self.outdoor_heat_coeff * (outdoor_temp - self.indoor_temp) * self.time_step
        delta_temp = (q_ac + q_outdoor) / self.room_heat_capacity
        self.indoor_temp += delta_temp
        self.indoor_temp = np.clip(self.indoor_temp, 10.0, 40.0)  # Constrain indoor temperature

        # reward calculation
        reward = self.calculate_reward()

        self.history.append([self.indoor_temp, outdoor_temp])
        self.last_temp = self.indoor_temp

        self.current_step += 1
        done = self.current_step >= len(self.historical_temps)

        return self._get_obs(), reward, done, {}

    def calculate_reward(self):
        # take energy efficiency and indoor temperature stability into account
        temp_error = abs(self.indoor_temp - self.target_temp)
        stability = np.exp(-0.5 * (self.indoor_temp - self.last_temp)**2)
        energy_cost = self.ac_power * self.time_step

        # set weighting
        temp_weight = 1.0 if temp_error > 2 else 0.5
        reward = (
            -temp_weight * temp_error
            + 0.5 * stability
            - 0.02 * energy_cost
        )
        return reward

    def _get_obs(self):
        # return the observed values of the past 10 days
        while len(self.history) < self.hist_len:
            self.history.appendleft(self.history[0])

        return np.array(self.history, dtype=np.float32)[-self.hist_len:]
from ac_env import ACControlEnv
import numpy as np

# test the environment class
historical_temps = np.random.uniform(15, 30, 20)  # mock historical temperature data
print(historical_temps.shape)
env = ACControlEnv(
    historical_temps=historical_temps,
    hist_len=10
)

# print initial states
obs = env.reset()
print("Initial observation data shape:", obs.shape)
print("Initial room temperature:", obs[-1, 0])

# print states after 1 action
action = 5  # i.e. 16+5=21°C
next_obs, reward, done, _ = env.step(action)
print("\nObaservation data shape after 1 action:", next_obs.shape)
print("New room temperature:", next_obs[-1, 0])
print("Value of reward:", reward)
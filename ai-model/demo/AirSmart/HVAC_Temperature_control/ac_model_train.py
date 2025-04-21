import pandas as pd
import numpy as np
from ac_env import ACControlEnv
from ac_model import Actor_Critic
import torch

# variable list
place = "HKA"
path = "" # path you store the data

# prepare training data
train_seq = pd.read_csv(path+"TRAIN_"+place+"_DATASET_.csv")
train_seq = np.array(train_seq["Mean Temp"])

# set environment
env = ACControlEnv(train_seq)

model = Actor_Critic(env)

results = {"reward":[], "actions":[]}

# train AC model
for episode in range(200):
    state = env.reset()
    episode_data = []
    ep_r = 0

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, entropy = model.get_action(state_tensor)
        next_state, reward, done, _ = env.step(action)
        act_temp = action + 16

        episode_data.append((
            state,
            (log_prob, entropy),
            reward,
            next_state,
            float(done)
        ))

        state = next_state
        ep_r += reward
        if done:
            break

    # convert to tensor
    states = [d[0] for d in episode_data]
    actions = [d[1] for d in episode_data]
    rewards = [d[2] for d in episode_data]
    next_states = [d[3] for d in episode_data]
    dones = [d[4] for d in episode_data]

    # save actions and rewards
    results["reward"].append(rewards)
    results["actions"].append(act_temp)

    # model learn
    model.learn_batch(states, actions, rewards, next_states, dones)
    print(f"episode: {episode}\tep_r: {ep_r}")
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from . import ac_env as A
from matplotlib import pyplot as plt

# define Actor Model
class Actor(nn.Module):
    def __init__(self, action_dim, state_dim=2):
        super().__init__()
        # As we wish to process temp data of past 10 days, we need lstm layer to capture time features
        self.lstm = nn.LSTM(input_size=state_dim,
                    hidden_size=64,
                    num_layers=1,
                    batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.4),
            nn.Linear(32, action_dim)
        )

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        out, (h_n, _) = self.lstm(s)
        action_probs = F.softmax(self.fc(out[:,-1,:]), dim=-1)
        return action_probs

# define Critic Model
class Critic(nn.Module):
    def __init__(self, state_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(state_dim,64,batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        out, _ = self.lstm(s)
        value = self.fc(out[:, -1, :])

        return value.squeeze(-1)

# build AC model
class Actor_Critic:
    def __init__(self, env, gamma=0.99, lr_a=3e-4, lr_c=5e-4):
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c

        # build AC model
        self.actor = Actor(env.action_space.n)
        self.critic = Critic()

        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer: Adam
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=200)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=200)

        # loss function: MSE
        self.loss = nn.MSELoss()

    def get_action(self, s):
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
        original_probs = self.actor(s)
        probs = original_probs.clone()
        current_outdoor_temp = s[0, -1, 1].item()
        if current_outdoor_temp > 28:
            mask = torch.zeros_like(probs)
            mask[:, :10] = 1.0  # allow temp setting lower than 26°C
        elif current_outdoor_temp < 10:
            mask = torch.zeros_like(probs)
            mask[:, 5:] = 1.0   # allow temp setting higher than 20°C
        else:
            mask = torch.ones_like(probs)

        probs = (probs * mask) / (probs.sum(dim=-1, keepdim=True) + 1e-8)  # normalization
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy()

    def learn_batch(self, states, action, reward, next_states, dones, ent_coef=0.01, tau=0.01):
        # conversion to tensor
        state_tensors = torch.FloatTensor(np.array(states))
        next_state_tensors = torch.FloatTensor(np.array(next_states))
        rewards_tensor = torch.FloatTensor(reward)
        dones_tensor = torch.FloatTensor(dones)
        log_probs = torch.stack([a for a, _ in action])
        entropies = torch.stack([e for _, e in action])

        # calculate critic value
        with torch.no_grad():
            target_values = self.critic_target(next_state_tensors)
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * target_values.squeeze()

        # calculate loss
        current_values = self.critic(state_tensors).squeeze()
        critic_loss = F.mse_loss(current_values, targets)

        # update critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # calculate Advantage
        advantages = (targets - current_values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # calculate loss
        policy_loss = -(log_probs * advantages).mean() - ent_coef * entropies.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        self.actor_scheduler.step()
        self.critic_scheduler.step()

# define ac model tester class
class AC_Tester:
    def __init__(self, model_path, test_data, target_temp=22, hist_len=10):
        # set environment
        self.env = A.ACControlEnv(
            historical_temps=test_data,
            target_temp=target_temp,
            hist_len=hist_len
        )

        self.hist_len = hist_len

        #  load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Actor_Critic(self.env)
        self._load_model(model_path)

        # save test results
        self.results = {
            'episode_rewards': [],
            'temperature_traces': [],
            'action_sequences': [],
            'energy_consumptions': [],
            'outdoor_temperatures': []
        }

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor.eval()
        self.agent.critic.eval()
        print(f"Load model sucessfully: {model_path}")

    def run_test(self, num_episodes=5, render_interval=10):
        for ep in tqdm(range(num_episodes), desc="Testing progress"):
            state = self.env.reset()
            done = False
            ep_data = {
                'rewards': [],
                'indoor_temps': [],
                'actions': [],
                'outdoor_temps': [],
                'energy': 0.0
            }

            while not done:
                # tensor conversion
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # get action
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(state_tensor)
                    action_temp = 16 + action  # convert to temperature

                # take action
                next_state, reward, done, _ = self.env.step(action)

                # keep record
                ep_data['rewards'].append(reward)
                ep_data['indoor_temps'].append(self.env.indoor_temp)
                ep_data['actions'].append(action_temp)
                ep_data['outdoor_temps'].append(self.env.historical_temps[self.env.current_step-1])

                # update state
                state = next_state

            # energy calculation
            energy = []
            _energy = self.env.ac_capacity * abs(np.array(ep_data['outdoor_temps'])
                                                    - np.array(ep_data['indoor_temps'])) * self.env.time_step
            energy.append(list(_energy))

            # save results
            self.results['episode_rewards'].append(sum(ep_data['rewards']))
            self.results['temperature_traces'].append(ep_data['indoor_temps'])
            self.results['action_sequences'].append(ep_data['actions'])
            self.results['energy_consumptions'].append(energy[0])
            self.results['outdoor_temperatures'].append(ep_data['outdoor_temps'])

            # visualize result
            if (ep+1) % render_interval == 0:
                self._visualize_episode(ep)

        print()
        print("finish testing!")
        self._generate_report()
        return self.results

    def predict(self):
        state = self.env.reset()
        done = False
        ep_data = {
            'rewards': [],
            'indoor_temps': [],
            'actions': [],
            'outdoor_temps': [],
            'energy': 0.0
        }

        while not done:
            # tensor conversion
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # get action
            with torch.no_grad():
                action, _, _ = self.agent.get_action(state_tensor)
                action_temp = 16 + action  # convert to temperature

            # take action
            next_state, reward, done, _ = self.env.step(action)

            # keep record
            ep_data['rewards'].append(reward)
            ep_data['indoor_temps'].append(self.env.indoor_temp)
            ep_data['actions'].append(action_temp)
            ep_data['outdoor_temps'].append(self.env.historical_temps[self.env.current_step-1])

            # update state
            state = next_state

        # energy calculation
        energy = []
        _energy = self.env.ac_capacity * abs(np.array(ep_data['outdoor_temps'])
                                                 - np.array(ep_data['indoor_temps'])) * self.env.time_step
        energy.append(list(_energy))

        # save results
        self.results['episode_rewards'].append(sum(ep_data['rewards']))
        self.results['temperature_traces'].append(ep_data['indoor_temps'])
        self.results['action_sequences'].append(ep_data['actions'])
        self.results['energy_consumptions'].append(energy[0])
        self.results['outdoor_temperatures'].append(ep_data['outdoor_temps'])

        self._visualize_temp()
        self._visualize_actemp()

        self._generate_report()
        return self.results

    # visualize a single episode
    def _visualize_episode(self, ep_idx):
        fig, axs = plt.subplots(3, 1, figsize=(12, 9))

        # temperature
        axs[0].plot(self.results['temperature_traces'][ep_idx], label='Indoor Temperature', color='blue')
        axs[0].plot(self.results['outdoor_temperatures'][ep_idx], '--', label='Outdoor Temperature', color='orange')
        axs[0].axhline(self.env.target_temp, color='red', linestyle='--', label='Target Temperature')
        axs[0].set_title(f"Episode {ep_idx+1} - Temperature Control")
        axs[0].set_ylabel("Temperature (°C)")
        axs[0].legend()

        # set temperature
        axs[1].step(range(len(self.results['action_sequences'][ep_idx])),
                   self.results['action_sequences'][ep_idx],
                   where='post', color='green')
        axs[1].set_ylim(16, 30)
        axs[1].set_title("AC Temperature Setting")
        axs[1].set_ylabel("Temperature (°C)")

        # reward
        axs[2].plot(self.results['episode_rewards'][:ep_idx+1], marker='o')
        axs[2].set_title("Change of Reward")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Reward")

        plt.tight_layout()
        plt.show()

    # visualize environmental temperature for predict method
    def _visualize_temp(self):
        plt.figure(dpi=128,figsize=(10,6))

        # temperature
        plt.plot(self.results['temperature_traces'][0], label='Indoor Temperature', color='blue')
        plt.plot(self.results['outdoor_temperatures'][0], '--', label='Outdoor Temperature', color='orange')
        plt.axhline(self.env.target_temp, color='red', linestyle='--', label='Target Temperature')
        plt.title('Indoor and Outdoor Temperature After setting AirSmart',fontsize=20)
        plt.ylabel("Temperature (°C)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # visualize AC temperature setting for predict method
    def _visualize_actemp(self):
        plt.figure(dpi=128,figsize=(10,6))

        # AC temperature
        plt.step(range(len(self.results['action_sequences'][0])),
                   self.results['action_sequences'][0],
                   where='post', color='green')
        plt.ylim(16, 30)
        plt.title("AirSmart Temperature Setting", fontsize=20)
        plt.ylabel("Temperature (°C)", fontsize=12)
        plt.xlabel("Day", fontsize=12)

        plt.tight_layout()
        plt.show()

    # analyze performance of the model
    def _generate_report(self):
        avg_reward = np.mean(self.results['episode_rewards'])
        avg_energy = np.mean(self.results['energy_consumptions'])/self.hist_len
        temp_errors = [np.mean(np.abs(np.array(trace) - self.env.target_temp))
                      for trace in self.results['temperature_traces']]

        print("\n" + "="*40)
        print(f"{' Test report ':=^40}")
        print(f"No of testing episodes: {len(self.results['episode_rewards'])}")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average temperature error: {np.mean(temp_errors):.2f} ± {np.std(temp_errors):.2f} °C")
        print(f"Average energy consumption: {avg_energy:.2f} ± {np.std(self.results['energy_consumptions'])/self.hist_len:.2f} kW·h")
        print("="*40)

class AC_output:
    def __init__(self, model_path, hist_len=10, feature_dim = 2):
        self.hist_len = hist_len
        self.feature_dim = feature_dim
        self.actor = Actor(action_dim = 15)
        self.critic = Critic()

        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.eval()

    # predict the AC temperature setting
    def predict(self, raw_data):
        raw_data = self._get_full_data(raw_data)

        input_tensor = torch.FloatTensor(raw_data).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.actor(input_tensor)
            action = torch.argmax(action_probs).item()

        return 16 + action

    # fill the dataset if the data is shorter than the required length
    def _get_full_data(self, raw_data):
        while raw_data.shape[0] < self.hist_len:
            raw_data = np.concatenate((raw_data[0].reshape(1, -1), raw_data[:]), axis=0)

        return raw_data[-self.hist_len:]
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datetime import datetime
import itertools
import argparse
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data(path: str):
    if not path:
        raise ValueError('no path was specified')
    df = pd.read_csv('NOK.csv')
    return df['Close'].values.reshape(-1, 1)


## experience buffer

class ReplayBuffer:
    def __init__(self, state_dim, act_dim, size):
        self.cur_state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.size_max = size

    def store(self, state, act, reward, next_state, done):
        self.cur_state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size_max
        self.size = min(self.size + 1, self.size_max)

    def sample(self, batch_size=32):
        indices = np.random.randint(0, self.size, size=batch_size)
        return dict(s1=self.cur_state_buf[indices],
                    s2=self.next_state_buf[indices],
                    a=self.acts_buf[indices],
                    r=self.rews_buf[indices],
                    d=self.done_buf[indices]
                    )


class Utils:

    @classmethod
    def get_fit_scaler(cls, env, episodes=5):
        states = []
        for _episode in range(episodes):
            env.reset()
            for _step in range(env.n_step):
                action = np.random.choice(env.action_space)
                state, _, done, _, = env.step(action)
                states.append(state)
                if done:
                    break
        scaler = StandardScaler()
        scaler.fit(states)
        return scaler

    @classmethod
    def make_dir(cls, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

    @classmethod
    def predict(cls, model, inputs):
        with torch.no_grad():
            inputs = torch.from_numpy(inputs.astype(np.float32))
            outputs = model(inputs)
            return outputs.numpy()

    @classmethod
    def train_one_step(cls, model, criterion, optimizer, inputs, targets):
        inputs = torch.from_numpy(inputs.astype(np.float32))
        targets = torch.from_numpy(targets.astype(np.float32))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


class MLP(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden=32, n_layers=1):
        super(MLP, self).__init__()

        M = n_inputs
        self.layers = []
        for _ in range(n_layers):
            layer = nn.Linear(M, n_hidden)
            M = n_hidden
            self.layers.append(layer)
            self.layers.append(nn.ReLU())

        # final layer
        self.layers.append(nn.Linear(M, n_actions))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.layers(X)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class Env:
    def __init__(self, data, intial_investment=20000):
        self.asset_price_history = data
        self.n_step, self.n_asset = self.asset_price_history.shape
        self.intial_investment = intial_investment
        self.cur_step = None
        self.asset_owned = None
        self.asset_price = None
        self.remaining_cash = None
        self.action_space = np.arange(3 ** self.n_asset)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_asset)))
        self.state_dim = 2 * self.n_asset + 1
        self.reset()

    def reset(self):
        self.cur_step = 0
        self.asset_owned = np.zeros(self.n_asset)
        self.asset_price = self.asset_price_history[self.cur_step]
        self.remaining_cash = self.intial_investment
        return self._get_state()

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        self.cur_step += 1
        self.asset_price = self.asset_price_history[self.cur_step]

        self._trade(action)

        cur_val = self._get_val()

        reward = cur_val - prev_val

        done = self.cur_step == self.n_step - 1

        info = dict(cur_val=cur_val)

        return self._get_state(), reward, done, info

    def _get_state(self):
        state = np.zeros(self.state_dim)
        state[:self.n_asset] = self.asset_owned
        state[self.n_asset:2 * self.n_asset] = self.asset_price
        state[-1] = self.remaining_cash
        return state

    def _get_val(self):
        return self.asset_owned.dot(self.asset_price) + self.remaining_cash

    def _trade(self, action):

        actions_to_do = self.action_list[action]

        sell_index = []
        buy_index = []
        for index, action in enumerate(actions_to_do):
            if action == 0:
                sell_index.append(index)
            elif action == 2:
                buy_index.append(index)
        if sell_index:
            for index in sell_index:
                self.remaining_cash += self.asset_price[index] * self.asset_owned[index]
                self.asset_owned[index] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                # TODO: make more efficient
                for index in buy_index:
                    if self.remaining_cash >= self.asset_price[index]:
                        self.asset_owned[index] += 1
                        self.remaining_cash -= self.asset_price[index]
                    else:
                        can_buy = False


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = MLP(state_size, action_size)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        # return np.random.choice(self.action_size)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = Utils.predict(self.model, state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        # first check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return

        # sample a batch of data from the replay memory
        minibatch = self.memory.sample(batch_size)
        states = minibatch['s1']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        # Calculate the target: Q(s',a')
        target = rewards + (1 - done) * self.gamma * np.amax(Utils.predict(self.model, next_states), axis=1)

        # With the PyTorch API, it is simplest to have the target be the
        # same shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = Utils.predict(self.model, states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step
        Utils.train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
        state = next_state

    return info['cur_val']


if __name__ == '__main__':

    # config
    models_folder = 'rl_trader_models'
    rewards_folder = 'rl_trader_rewards'
    num_episodes = 500
    batch_size = 32
    initial_investment = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()

    Utils.make_dir(models_folder)
    Utils.make_dir(rewards_folder)

    data = get_data('stocks.csv')
    n_timesteps, n_stocks = data.shape




    n_train = n_timesteps // 2
    # print(n_train)

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = Env(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = Utils.get_fit_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == 'test':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = Env(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/dqn.ckpt')

    # play the game num_episodes times
    # vals = []
    # for e in range(num_episodes):
    #     t0 = datetime.now()
    #     val = play_one_episode(agent, env, args.mode)
    #     vals.append(val)
    #     dt = datetime.now() - t0
    #     print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    #     portfolio_value.append(val)  # append episode end portfolio value
    # print('Average return: {}'.format(np.mean(vals)))
    # print('Median return: {}'.format(np.median(vals)))
    # save the weights when we are done
    if args.mode == 'train':
        # save the DQN
        agent.save(f'{models_folder}/dqn.ckpt')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

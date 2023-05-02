import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

from evaluation import plot_trading_chart
from utils import Actions

# Adapted from https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/envs/trading_env.py


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._base_amount = 1.

        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._has_open_position = False
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._has_open_position = False
        self._action_history = (self.window_size * [None])
        self._total_reward = 0.
        self._total_profit = 0.
        self.history = {}

        return self._get_observation()

    def step(self, action):
        # TODO: Be careful if actions are enum or integer values!
        self._action_history.append(action)

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = 0
        step_profit = 0

        if (self._done or action == Actions.Sell) and self._has_open_position:
            start_price = self.prices[self._last_trade_tick]
            end_price = self.prices[self._current_tick]

            step_reward = end_price - start_price
            step_profit = self._base_amount * (end_price / start_price - 1)

            self._has_open_position = False
            self._last_trade_tick = self._current_tick

        elif action == Actions.Buy and not self._has_open_position:
            self._has_open_position = True
            self._last_trade_tick = self._current_tick

        self._total_reward += step_reward
        self._total_profit += step_profit

        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            has_open_position=self._has_open_position,
        )
        self._update_history(info)

        self._current_tick += 1

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self, mode='human'):
        plot_trading_chart(self.df, self._action_history)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        return

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        signal_features = self.df

        return prices, signal_features

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from evaluation import plot_trading_chart
from utils import Actions

# only for testing
from datetime import datetime
from alpaca.data.timeframe import TimeFrame
from data import get_crypto_bars 


# Adapted from https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/envs/trading_env.py


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, time_unit="D"):
        assert df.ndim == 2

        self.seed()
        self.df = df # has as many entries as there are time steps in this environment
        self.portfolio = {}
        self.portfolio["cash"] = np.zeros((len(df), 1))
        self.initial_cash = 100
        self.portfolio["cash"][:window_size+1] = self.initial_cash
        self.portfolio["pos_units"] = np.zeros((len(df), 1))
        self.portfolio["pos_val"] = np.zeros((len(df), 1))
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
        self.time_unit = time_unit

        self._encoder = None
        self._encoder_exists = False

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
        
        self._encoder = None
        self._encoder_exists = False

        return self._get_observation()


    def step(self, action):
        # TODO: Be careful if actions are enum or integer values!
        self._action_history.append(action)

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = 0
        step_profit = 0

        # sell all?
        if (self._done or action == Actions.Sell) and self._has_open_position:
            start_price = self.prices[self._last_trade_tick]
            end_price = self.prices[self._current_tick]

            step_reward = end_price - start_price # TODO is this correct? (base_amount/start_price)*end_price) - base_amount/start_price*start_price )
            step_profit = self._base_amount * (end_price / start_price - 1)
            
            self.portfolio["cash"][self._current_tick] += self.portfolio["pos_units"] * self.prices[self._current_tick]
            self.portfolio["pos_units"][self._current_tick] = 0

            self._has_open_position = False
            self._last_trade_tick = self._current_tick

        # buy for USD base_amount equivalent
        elif action == Actions.Buy and not self._has_open_position:
            self._has_open_position = True
            self._last_trade_tick = self._current_tick
            self.portfolio["cash"][self._current_tick] -= self._base_amount
            self.portfolio["pos_units"][self._current_tick] += self._base_amount / self.prices[self._current_tick]

        self.portfolio["pos_val"][self._current_tick] = self.portfolio["pos_units"][self._current_tick] * self.prices[self._current_tick]


            # reward???

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

        self.portfolio["cash"][self._current_tick] = self.portfolio["cash"][self._current_tick-1]
        self.portfolio["pos_units"][self._current_tick] = self.portfolio["pos_units"][self._current_tick-1]
        #self.portfolio["pos_val"][self._current_tick] = self.portfolio["pos_val"][self._current_tick-1]


        return observation, step_reward, self._done, info

    def get_encoder(self):
        if not self._encoder_exists:
            self._encoder = Encoder()
            self._encoder_exists = True
        return self._encoder

    def _get_observation(self):
        pos_data = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        # deliberately not including total profit because shouldnt affect trading behavior
        port_data = np.squeeze(np.array([val[(self._current_tick-self.window_size+1):self._current_tick+1] for val in self.portfolio.values()])).T # [window, features] 
        pos_chart = plot_trading_chart(pos_data, ret_img=True) # [h, w, 1]
        # convert dates to day delta to first date
        pos_data['timestamp'] = (pos_data['timestamp'] - pos_data['timestamp'].min()) / np.timedelta64(1,self.time_unit)
        #print(pos_data.dtypes)
        pos_data = pos_data.to_numpy() # [window, cols]
        print(port_data.shape, port_data[-5:])
        print(pos_data.shape, pos_data[-5:])
        fin_data = np.concatenate([pos_data, port_data], axis=-1) # [window, cols+features]
        obs = self.get_encoder().encode(fin_data, pos_chart)
        return obs

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
    


class Encoder():
    def __init__(self) -> None:
        self._generate_NN()
        self._generate_img_compressor()
        self.trans = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(240),
                                        transforms.ToTensor()])

    def _generate_NN(self):
        self.generator = NN(11,12)

    def _generate_img_compressor(self):
        self.img_encoder = CNN()

    def train(self):
        pass

    def encode(self, fin_data, fin_chart_img, transform_img=True):
        if transform_img:
            input_img = self.trans(fin_chart_img)
            input_img = torch.unsqueeze(input_img,0) # [1, 1, h, w]
        else: 
            input_img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fin_chart_img),0),0)
        img_encoded = self.img_encoder(input_img) # [1, output_dim_img]
        # a) transformer on all
        #fin_data = torch.unsqueeze(torch.flatten(torch.from_numpy(fin_data)),0) # [1, window*cols]
        #input = torch.concat([fin_data, img_encoded], dim=-1) # [1, (window*cols)+input_img]
        #data_encoded = self.generator(input)
        # b) transformer only on windowed-data and cnn separate
        fin_data = torch.unsqueeze(torch.from_numpy(fin_data),0).float() # [1, window, cols]
        fin_encoded = self.generator(fin_data) # # [1, output_dim_transformer]
        data_encoded = torch.concat([fin_encoded, img_encoded], dim=-1) # [1, output_dim_transformer+output_dim_img]
        print("encoded shape", data_encoded.shape)
        return data_encoded



class NN(torch.nn.Module):
    def __init__(self, in_size, out_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=in_size,
                    nhead=1,
                    batch_first=True
                ),
                num_layers = 3
            ),
            nn.Flatten(start_dim=1), # batch is retained, sequence not
            nn.LazyLinear(out_size),
            #nn.ReLU()
        ])
    
    # idea: train self-supervised as autoencoder
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output
    
    def forward_inference(self, input):
        pass


class CNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            nn.Conv2d(1,8,3, stride=2), # expect b/w inputs
            nn.ReLU(),
            nn.Conv2d(8,32,3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32,96,3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(12)
        ])
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

if __name__ == '__main__':
    enc = Encoder()
    test = torch.rand((3,5,11))
    out = enc.generator(test)
    print(out.shape)#, torch.all(out==test), torch.all(out))

    cnn = CNN()
    out = torch.rand((1,1,240,240))
    for layer in cnn.layers:
        out = layer(out)
        print(out.shape)
    #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)

    bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 1),
                       datetime(2022, 7, 1), timeframe=TimeFrame.Day)
    env = TradingEnv(bars, window_size=10)
    env.reset()
    for i in range(20):
        print(i, env.step(Actions.Buy))
        
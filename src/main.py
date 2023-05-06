from datetime import datetime

from alpaca.data.timeframe import TimeFrame
from matplotlib import pyplot as plt

from data import get_crypto_bars
from experts import expert_1, expert_2, get_expert_trajectories
from stock_env import TradingEnv

bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 1),
                       datetime(2022, 7, 1), timeframe=TimeFrame.Day)
expert_actions = expert_2(bars)

print(bars.head())
print(expert_actions[:5])

window_size = 10
env = TradingEnv(bars, window_size)

observation = env.reset()
tick = window_size
while True:
    action = expert_actions[tick]
    observation, reward, done, info = env.step(action)
    if done:
        print("info:", info)
        break

    tick += 1

env.render_all()
#plt.show()

get_expert_trajectories()

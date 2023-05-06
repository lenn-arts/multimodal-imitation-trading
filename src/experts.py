from datetime import datetime

import numpy as np
from alpaca.data.timeframe import TimeFrame
from stock_env import TradingEnv

from data import get_crypto_bars
from utils import Actions
from imitation.data import types
from evaluation import plot_trading_chart


def expert_1(bars):
    """
    A simple expert strategy that recommends to buy if the next day's closing price is higher
    than the current day's closing price, and sell otherwise.

    Args:
        bars (pd.DataFrame): A DataFrame containing historical price data with 'close' column.

    Returns:
        np.ndarray: An array of 'buy' and 'sell' actions based on the simple strategy.
    """
    return np.where(bars['close'].shift(-1) > bars['close'], Actions.Buy, Actions.Sell)


def expert_2(bars):
    """
    An expert strategy that recommends to buy when the next day's closing price is going up,
    hold while the price continues to go up, sell as soon as the price starts going down,
    hold while the price continues to go down, and sell at the end of the DataFrame if still
    holding a position.

    Args:
        bars (pd.DataFrame): A DataFrame containing historical price data with 'close' column.

    Returns:
        list: A list of 'buy', 'hold', and 'sell' actions based on the specified strategy.
    """
    n = len(bars)
    expert_actions = [Actions.Hold] * n
    holding_position = False

    for i in range(n - 1):
        if bars['close'][i + 1] > bars['close'][i] and not holding_position:
            expert_actions[i] = Actions.Buy
            holding_position = True
        elif bars['close'][i + 1] < bars['close'][i] and holding_position:
            expert_actions[i] = Actions.Sell
            holding_position = False

    # Sell at the end of the dataframe if still holding a position
    if holding_position:
        expert_actions[-1] = Actions.Sell

    return expert_actions

def action2int(action):
    if action == Actions.Sell: out = 0
    if action == Actions.Buy: out = 1
    if action == Actions.Hold: out = 2
    return out

def get_expert_trajectories(num_trajs=1):
    """ generate trajectories for IRL algo.
      A trajectory is a TrajectoryWithRew object holding 
      observations: np.array
      actions: np.array
      infos: None,
      terminal: boolean
      rewards: np.array"""
    # data only reaches back to 2021-01-01
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ALGO/USD"]
    symbols_dict = {i:symbols[i] for i in range(len(symbols))}
    
    bars_org = {val: get_crypto_bars(val, datetime(2021, 1, 1),
                       datetime(2023, 4, 1), timeframe=TimeFrame.Day) for key, val in symbols_dict.items()}
    trajectories = []
    ws = 50 
    max_steps = 365
    min_steps = ws + 20
    print("bars_org", bars_org.keys(), len(bars_org["BTC/USD"]))
    for i_traj in range(num_trajs):
        symbol = np.random.choice(np.array(list(range(len(symbols_dict.values())))), size=1)[0]
        symbol = symbols_dict[symbol]
        length = len(bars_org[symbol])
        print("chose", symbol)
        indices = np.random.choice(
            np.array(list(range(length))), size=2, replace=False)
        begin = indices.min()
        # a)
        #end = indices.max()
        # b)
        end = begin + np.random.choice(np.array(list(range(min_steps, max_steps))), size=1, replace=False)[0]
        print("begin", begin, "end", end)
        if end - begin <= ws: end += ws # if range too small
        if end > length: end = length
        bars = bars_org[symbol].copy(deep=True).iloc[begin:end+1].reset_index(drop=True)
        print(bars)
        env = TradingEnv(bars, ws)
        print("env created")
        expert = expert_2(bars)
        print("expert created")
        expert = expert[ws:] # TODO
        #plot_trading_chart(bars[ws:], actions=expert)
        print(len(expert))
        
        i_observations = []
        i_rewards = np.array([])
        i_done = False
        i_info = None

        obs = env.reset() # this set start tick to bars[0+window]
        i_observations.append(obs)#.detach().numpy())
        for j, action in enumerate(expert):
            #print("action", j)
            obs, reward, done, info = env.step(action)
            i_observations.append(obs)#.detach().numpy())
            i_rewards = np.append(i_rewards, reward)
            i_done = done
        i_observations = np.stack(i_observations, 0)
        print("obs:", i_observations.shape)
        print("rews:", i_rewards.shape)
        trajectories.append(types.TrajectoryWithRew(
            i_observations, np.array([action2int(action) for action in expert]), i_info, i_done, i_rewards
        ))
    return trajectories




if __name__ == '__main__':
    #bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 1),
    #                       datetime(2022, 7, 1), timeframe=TimeFrame.Day)
    #bars["expert_1"] = expert_1(bars)
    #bars["expert_2"] = expert_2(bars)
    #print(bars.head())
    get_expert_trajectories(num_trajs=5)

    bars = get_crypto_bars("BTC/USD", datetime(2021, 1, 1),
                       datetime(2023, 4, 1), timeframe=TimeFrame.Day)
    bars = bars[:10]
    ws = 5 
    env = TradingEnv(bars, ws)
    expert = expert_2(bars)
    expert = expert # TODO
    #plot_trading_chart(bars, actions=expert)

from datetime import datetime

import numpy as np
from alpaca.data.timeframe import TimeFrame

from data import get_crypto_bars


def expert_1(bars):
    """
    A simple expert strategy that recommends to buy if the next day's closing price is higher
    than the current day's closing price, and sell otherwise.

    Args:
        bars (pd.DataFrame): A DataFrame containing historical price data with 'close' column.

    Returns:
        np.ndarray: An array of 'buy' and 'sell' actions based on the simple strategy.
    """
    return np.where(bars['close'].shift(-1) > bars['close'], 'buy', 'sell')


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
    expert_actions = ['hold'] * n
    holding_position = False

    for i in range(n - 1):
        if bars['close'][i + 1] > bars['close'][i] and not holding_position:
            expert_actions[i] = 'buy'
            holding_position = True
        elif bars['close'][i + 1] < bars['close'][i] and holding_position:
            expert_actions[i] = 'sell'
            holding_position = False

    # Sell at the end of the dataframe if still holding a position
    if holding_position:
        expert_actions[-1] = 'sell'

    return expert_actions


if __name__ == '__main__':
    bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 1),
                           datetime(2022, 7, 1), timeframe=TimeFrame.Day)
    bars["expert_1"] = expert_1(bars)
    bars["expert_2"] = expert_2(bars)
    print(bars.head())

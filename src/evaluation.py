from datetime import datetime

import matplotlib.pyplot as plt
from alpaca.data.timeframe import TimeFrame

from data import get_crypto_bars
from experts import expert_1, expert_2
from utils import Actions


def plot_trading_chart(bars, actions):
    """
    Generate a chart with the close price at each timestamp and mark each buy action
    with a green upward triangle and each sell action with a red downward triangle.

    Args:
        bars (pd.DataFrame): A DataFrame containing historical price data with 'timestamp' and 'close' columns.
        actions (list): A list of actions ('buy', 'hold', 'sell') based on a trading strategy.
    """
    bars_with_actions = bars.copy()
    bars_with_actions['expert_action'] = actions

    buy_actions = bars_with_actions[bars_with_actions['expert_action'] == Actions.Buy]
    sell_actions = bars_with_actions[bars_with_actions['expert_action']
                                     == Actions.Sell]

    plt.figure(figsize=(15, 7))
    plt.plot(bars_with_actions['timestamp'],
             bars_with_actions['close'], label='Close Price', linewidth=1)

    plt.scatter(buy_actions['timestamp'], buy_actions['close'],
                marker='^', color='g', label='Buy')
    plt.scatter(sell_actions['timestamp'], sell_actions['close'],
                marker='v', color='r', label='Sell')

    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.title('Trading Chart')
    plt.legend()
    plt.grid()


def calculate_profit(bars, actions, base_amount=1):
    """
    Calculate the profit based on the historical price data and the given strategy actions,
    always buying a fixed amount and selling all current holdings.

    Args:
        bars (pd.DataFrame): A DataFrame containing historical price data with 'close' column.
        actions (list): A list of actions ('buy', 'hold', 'sell') based on a trading strategy.
        base_amount (float, optional): The fixed amount of money to spend on each buy action.
                                        Default is 100.

    Returns:
        float: The profit gained by following the strategy.
        list: A list of executed trades with buy and sell timestamps and the profit for each trade.

    """
    total_profit = 0
    holding_position = False
    btc_held = 0
    entry_timestamp = None
    executed_trades = []

    for i in range(len(actions)):
        if actions[i] == Actions.Buy and not holding_position:
            entry_timestamp = bars['timestamp'][i]
            btc_held = base_amount / bars['close'][i]
            holding_position = True
        elif actions[i] == Actions.Sell and holding_position:
            trade_profit = btc_held * bars['close'][i] - base_amount
            total_profit += trade_profit
            executed_trades.append({
                'buy_timestamp': entry_timestamp,
                'sell_timestamp': bars['timestamp'][i],
                'holding_duration': bars['timestamp'][i] - entry_timestamp,
                'profit': trade_profit
            })
            holding_position = False
            btc_held = 0

    return total_profit, executed_trades


def print_summary(profit, trades):
    print(f"Profit: {profit}")
    print(f"Number of trades: {len(trades)}")
    avg_holding_duration = sum(
        [trade['holding_duration'].days for trade in trades]) / len(trades)
    print(f"Average holding duration in days: {avg_holding_duration}")
    avg_profit = sum([trade['profit'] for trade in trades]) / len(trades)
    print(f"Average profit per trade: {avg_profit}")


if __name__ == '__main__':
    bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 11),
                           datetime(2022, 7, 1), timeframe=TimeFrame.Day)

    actions = expert_2(bars)
    profit, trades = calculate_profit(bars, actions)
    print("Expert 2")
    print_summary(profit, trades)
    plot_trading_chart(bars, actions)
    plt.show()

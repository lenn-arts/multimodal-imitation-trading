import os
from datetime import datetime

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


def get_crypto_bars(symbol, start, end, timeframe=TimeFrame.Day, limit=None):
    # Generate the file name based on the provided parameters
    file_name = f"{symbol.replace('/', '-')}_{start}_{end}_{timeframe}"
    if limit is not None:
        file_name += f"_limit{limit}"
    file_name += ".csv"
    file_path = os.path.join("data", file_name)

    # Check if the data already exists in the data/ folder
    if os.path.isfile(file_path):
        # If it exists, read it
        print("Using cached data")
        bars = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        # If it doesn't exist, download it using the get_crypto_bars function
        print("Downloading data")
        bars = download_data(symbol, start, end, timeframe, limit)

        # Save the data to the data/ directory
        os.makedirs("data", exist_ok=True)
        bars.to_csv(file_path)

    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    return bars


def download_data(symbol, start, end, timeframe=TimeFrame.Day, limit=None):
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit
    )
    bars = client.get_crypto_bars(request_params).df
    bars.reset_index(inplace=True)
    bars.drop(columns=['symbol'], inplace=True)
    return bars


if __name__ == '__main__':
    bars = get_crypto_bars("BTC/USD", datetime(2021, 7, 1),
                           datetime(2022, 7, 1), timeframe=TimeFrame.Day)
    print(bars.head())
    print(bars.dtypes)

from binance.client import Client
import pandas as pd

# use API from Binance
api_key = 'prpbKfyqt9yegycxxqnUMRTlLJbTvetR2BiNRoyiDbp8AyLnWdfvnjSgosbX5tQM'
api_secret = 'sNf8wX1DLLzLlx0MC9dj5obr03fCULB978HvCi3L9rCdaxspfbi5HlYlPj92nZOB'

# initialise client
client = Client(api_key, api_secret)


# get historical k-lines（BTC/USDT）
def get_historical_klines(symbol, interval, start_date, end_date=None):
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)

    # transform to dataframe
    df = pd.DataFrame(klines, columns=[
        'Date', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # formatting timestamp
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df['volatility'] = df['close'].rolling(window=24).std()
    df.dropna(inplace=True)

    # keep key data
    dataset = df[['close', 'volume', 'volatility']].astype(float)
    return dataset


# get BTC/USDT data from 2024-01-09 to 2025-01-09
symbol = 'BTCUSDT'
interval = '1h'
start_date = '2024-01-07'
end_date = '2025-01-10'
data = get_historical_klines(symbol, interval, start_date, end_date)

# save as a csv file
data.to_csv('btc_usdt.csv')
print(data.tail())

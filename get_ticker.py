# to import up-to-date bitcoin prices
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
def get_bitcoin_prices():
    symbol = 'BTC-USD'
    end_date = datetime.now() - timedelta(days=1)
    bitcoin_data = yf.download(symbol, start="2018-2-1", end=end_date)
    return bitcoin_data

'''
The index is divided into the following four categories:
0–24: Extreme fear (orange)
25–49: Fear (amber/yellow)
50–74: Greed (light green)
75–100: Extreme greed (green)
'''

# import Crypto greed fear index
def get_greed_index():
    r = requests.get('https://api.alternative.me/fng/?limit=0')
    data_points = r.json()['data'] #only need data portion of the json
    # create a DataFrame with multiple columns
    gi_df = pd.DataFrame(data_points, columns=['timestamp', 'value', 'value_classification'])
    return gi_df


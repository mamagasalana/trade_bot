import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import math
import glob
import json
import re
from src.maps.config import *
import src.maps.economic_classification as econ_class
from  src.maps.economic_classification import *

class reader:
    def __init__(self, ccy='EURUSD'):
        self.ccy = None #active currency
        self.ccys = {}
        self._fxs = {}
        self._chart_fxs = {}
        self.instrument_sentiment = {}
        self.INSTRUMENTS= json.load(open('src/maps/INSTRUMENT_MAP.json'))

        self.load_currency(ccy)
        self.load_etoro()
        self.load_forexfactory()

    @property
    def fx(self):
        return self.ccys[self.ccy]
    
    @property
    def fxs(self):
        if not self.ccy in self._fxs:
            self._fxs[self.ccy] = {}
        return self._fxs[self.ccy]
    
    @property
    def chart_fxs(self):
        if not self.ccy in self._chart_fxs:
            self._chart_fxs[self.ccy] = {}
        return self._chart_fxs[self.ccy]
    
    def query_ff(self, start_date=None, end_date=None):
        if start_date is None or end_date is None:
            ret1 = self.ff[(self.ff.currency==self.ccy[:3])].copy()
            ret2 = self.ff[(self.ff.currency==self.ccy[3:])].copy()
        else:
            ret1 = self.ff[(self.ff.currency==self.ccy[:3]) & (self.ff.datetime >= start_date) & (self.ff.datetime <= end_date)].copy()
            ret2 = self.ff[(self.ff.currency==self.ccy[3:]) & (self.ff.datetime >= start_date) & (self.ff.datetime <= end_date)].copy()
        return ret1, ret2
    
    def event_metadata(self):
        out = []
        for ret in self.query_ff():
            event_min = ret.groupby('event')['datetime'].min()
            event_max = ret.groupby('event')['datetime'].max()
            merged = pd.concat([event_min, event_max], axis=1)
            merged.columns = ['min', 'max']
            out.append(merged)
        return out
            
    def load_currency(self, ccy):
        self.ccy = ccy # set active
        if not ccy in self.ccys:
            fx= pd.read_csv(f'files/ccy/{ccy}.csv',  names =['dt', 'tm', 'Open', 'High', 'Low', 'Close', 'Volume'])
            fx['Date'] = pd.to_datetime(fx['dt'] + ' ' + fx['tm'])

            fx.replace([np.inf, -np.inf], np.nan, inplace=True)
            fx.dropna(inplace=True)
            fx.reset_index(inplace=True)
            fx = fx.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.ccys[ccy] = fx
        return self.ccys[ccy]

    def load_forexfactory(self):
        specs = []
        js = json.load(open('src/maps/forexfactory_eventid.json'))
        for k, v in js.items():
            out = {v2['title']: v2['html'] for v2 in v}
            out['eventid'] = int(k)
            specs.append(out)
        df_specs = pd.DataFrame(specs)
        df_specs = df_specs[~df_specs['FF Notes'].str.contains("discontinue", na=False)].copy()

        df=  pd.read_csv(FILES)
        df['time'] = df.time.fillna(method='ffill')
        df = df.query("actual.notnull() and time.str.contains('am|pm', na=False)").copy()
        df.loc[:, 'datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %I:%M%p')

        module_contents = [x for x in dir(econ_class) if not x.startswith('__')]
        econ_classification = { y:x  for x in module_contents for y in eval(x)}
        df.loc[:, 'eclass']  =df.event.map(econ_classification)
        self.ff = pd.merge(df, df_specs, on='eventid', how='inner')

    def load_etoro(self):
        
        for f in sorted(glob.glob('files/etoro/*.json')):
            dt = re.findall(r'\d+', f)[0]
            with open(f, 'r') as ifile:
                s = ifile.read()
            if not s.endswith(']'):
                idx = s.rfind("}")
                s = s[:idx+1] + ']'

            js = json.loads(s)
            for info in js:
                instrument_id = info['instrumentId']
                if not instrument_id in self.instrument_sentiment:
                    self.instrument_sentiment[instrument_id] = []
                info['date'] = dt
                self.instrument_sentiment[instrument_id].append(info)


    def resample(self, tf):
        """
        # '15T': 15-minute intervals.
        # '30T': 30-minute intervals.
        # '4H': 4-hour intervals.
        # '1D': : Daily intervals.
        """
        if not tf in self.fxs:    
            self.fxs[tf] = self.fx.resample(tf).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        return self.fxs[tf]
    
    def chart_fx(self, tf=None):
        if not tf in self.chart_fxs:
            if tf is not None:
                self.chart_fxs[tf] = self.resample(tf)
            else:
                self.chart_fxs[tf] = self.fx
        return self.chart_fxs[tf]

    def visualize_peak(self, idx, chart_size= 50, tf=None):
        if idx < chart_size or idx + chart_size > len( self.chart_fx(tf)):
            return
        marker_price = self.chart_fx(tf).iloc[idx]['High']

        fig, axlist = mpf.plot(self.chart_fx(tf).iloc[idx-chart_size:idx+chart_size+1], type='candle', style='charles', volume=True, 
                            title='OHLC Chart with Marker', ylabel='Price', ylabel_lower='Volume', 
                            returnfig=True)

        axlist[0].plot(chart_size, marker_price, 'x', color='red', markersize=12, label='Marker at x')
        plt.show()

    def visualize_2peak(self, idx, idx2, chart_size= 50, tf=None):
        if idx < chart_size or idx2 + chart_size > len( self.chart_fx(tf)):
            print(idx, chart_size)
            print(idx2, len( self.chart_fx(tf)))
            return
        marker_price = self.chart_fx(tf).iloc[idx]['High']
        marker_price2 = self.chart_fx(tf).iloc[idx2]['High']
        
        fig, axlist = mpf.plot(self.chart_fx(tf).iloc[idx-chart_size:idx2+chart_size+1], type='candle', style='charles', volume=True, 
                            title='OHLC Chart with Marker', ylabel='Price', ylabel_lower='Volume', 
                            returnfig=True)

        new_idx = chart_size
        new_idx2 = chart_size + (idx2- idx)
        slope = (marker_price2 - marker_price) / (idx2 - idx)
        # y = mx + b
        # marker_price =  slope * chart_size + b
        
        b =  marker_price - slope* chart_size
        c = slope* (new_idx2 + chart_size) + b

        # to maintain chart size
        new_b = self.chart_fx(tf).iloc[idx-chart_size:idx2+chart_size+1].High.max()
        new_b = math.ceil(new_b * 1000) / 1000
        new_c = self.chart_fx(tf).iloc[idx-chart_size:idx2+chart_size+1].Low.min()
        new_c = math.floor(new_c * 1000) / 1000
        
        new_x2 =  (new_c - b )//slope
        new_c =  slope*new_x2 + b
        x2 = new_idx2 + chart_size
        if c < new_c:
            c = new_c
            x2 = new_x2

        # new_b  =  slope * new_x + b
        new_x =  (new_b - b )//slope
        new_b =  slope*new_x + b
        x = 0
        if b  > new_b:
            b = new_b
            x = new_x

        axlist[0].plot([x, x2], [b, c], color='blue', linewidth=2, linestyle='--', label='Diagonal Line')
        axlist[0].plot(new_idx, marker_price, 'x', color='red', markersize=12, label='Marker at x')
        axlist[0].plot(new_idx2, marker_price2, 'x', color='red', markersize=12, label='Marker at x')
        plt.show()


    def visualize_sentiment(self):
        instrumentid = self.INSTRUMENTS.get(self.ccy)
        df = self.resample('1D').copy()
        dfx =pd.DataFrame(self.instrument_sentiment[instrumentid])
        dfx['date'] = pd.to_datetime(dfx['date'], format='%Y%m%d')

        dfx.set_index('date', inplace=True)
        df2 = df[df.index.isin(dfx.index)]
        dfx2 = dfx[dfx.index.isin(df2.index)]

        fig, axlist = mpf.plot(df2, type='candle', style='charles', volume=True, 
                            title='OHLC Chart with Sentiment', ylabel='Price', ylabel_lower='Volume', 
                            returnfig=True, figscale=1.5, figratio=(16, 9))

        ax_secondary = axlist[0].twinx()  # Create a secondary y-axis

        # Plot sentiment data on the secondary y-axis
        ax_secondary.plot( list(range(len(df2.index))), dfx2.buy , color='purple', linestyle='--', label='Buy Sentiment')
        ax_secondary.set_ylabel('Buy Sentiment')
        ax_secondary.spines['right'].set_color('purple')  # Color the secondary axis to match the data
        ax_secondary.tick_params(axis='y', colors='purple')

        # Display legend for the secondary y-axis
        ax_secondary.legend(loc='upper left')

if __name__ == '__main__':

    from scipy.signal import find_peaks
    r = reader('EURUSD')
    r.ff.to_csv('test2.csv')
    print('debug')
    # df = r.fx
    # arr, info = find_peaks(df.High, prominence=0.003)
    # arr2 = [x for x in arr if x > 2000 ]
    
    # for x, y in zip(arr2, arr2[1:]):
    #     if df.High.iloc[x] >  df.High.iloc[y]:
    #         r.visualize_2peak( x, y, 2000)
            


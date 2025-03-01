import requests
import pandas as pd
import os
import re
from src.csv.fred import FRED
import glob
import itertools

class BONDS_investingcom:
    def __init__(self):
        self.path = 'files/bond2/'
        self.flag_map = {
            'AUD' : 'AU',
            'CAD' : 'CA',
            'JPY' : 'JP',
            'USD' : 'US',
            'GBP' : 'GB',
            'EUR' : 'DE',
            'NZD' : 'NZ',
            'CHF' : 'CH'
            }   

    def get(self, ccy: list):
        dfmain = pd.DataFrame()
        for pair in ccy:
            pairmap = self.flag_map[pair]
            for f in glob.glob(self.path + '*%s*.csv' % pairmap ):
                df = pd.read_csv(f)
                fname = re.findall('bond2/(.*)\.csv', f)[0]
                fname= fname.replace(pairmap, pair)
                df = df.rename(columns={
                    'date' : 'datetime',
                    'o' : 'o_%s' % fname,
                    'h' : 'h_%s' % fname,
                    'l' : 'l_%s' % fname,
                    'c' : 'c_%s' % fname,
                })
                if dfmain.empty:
                    dfmain = df.copy()
                else:
                    dfmain = dfmain.merge(df, how='outer', on='datetime')

            df =  dfmain.set_index('datetime')
        
        return df
    
    def get_diff(self, ccy:list):
        df = self.get(ccy)
        cols = [ x for x in df.columns if x.startswith('c_')]
        tasks = list(itertools.combinations(cols, 2))
        dfs = []
        for col1, col2 in tasks:
            tmp = df[col1] - df[col2]
            tmp.name = '%s_%s' % (col1, col2)
            dfs.append(tmp)
        return pd.concat(dfs, axis=1)
            

class BONDS:
    def __init__(self):
        self.session = requests.Session()
        self.path = 'files/bond/'

    @property
    def CHF(self):
        fname = os.path.join(self.path, 'CHF.csv')
        url = 'https://data.snb.ch/api/cube/rendoblid/data/json/en'
        if not os.path.exists(fname):
            r = self.session.get(url)
            js = r.json()['timeseries']
            dfs= {}
            for itm in js:
                metadata = itm['metadata']
                lbl = re.findall('\{(.*?)\}', metadata['key'])
                new_lbl = lbl + ['CHF', 'BOND']
                new_name = '_'.join(new_lbl)
                for v in itm['values']:
                    dt = v['date']
                    if not dt in dfs:
                        dfs[dt] ={'datetime': dt }

                    dfs[dt][new_name] = v['value']
            
            df = pd.DataFrame(dfs.values()).sort_values('datetime').set_index('datetime')
            df.to_csv(fname)
            
        return pd.read_csv(fname)
    
    @property
    def USD(self):
        fname = os.path.join(self.path, 'USD.csv')
        if not os.path.exists(fname):
            fr = FRED(end_year=2025,start_year=2025)
            df = fr.get_tag('daily;h15')
            dfpivot = df.pivot(columns='event', index='datetime', values='actual')
            dfpivot.to_csv(fname)
        return pd.read_csv(fname)
    
    @property
    def CAD(self):
        fname = os.path.join(self.path, 'CAD.csv')
        if not os.path.exists(fname):
                
            url = 'https://www.bankofcanada.ca/valet/observations/CDN.AVG.1YTO3Y.AVG,CDN.AVG.3YTO5Y.AVG,CDN.AVG.5YTO10Y.AVG,CDN.AVG.OVER.10.AVG,BD.CDN.2YR.DQ.YLD,BD.CDN.3YR.DQ.YLD,BD.CDN.5YR.DQ.YLD,BD.CDN.7YR.DQ.YLD,BD.CDN.10YR.DQ.YLD,BD.CDN.LONG.DQ.YLD,BD.CDN.RRB.DQ.YLD/json'
            r = self.session.get(url)
            js = r.json()
            out=  []
            for itm in js['observations']:
                ret = {'datetime' : itm['d']}
                for k, v in itm.items():
                    if isinstance(v, dict) :
                        ret[k] = v['v']
                out.append(ret)
            df = pd.DataFrame(out)
            rename_dict ={}
            for k,v in js['seriesDetail'].items():
                lbl = v['label']
                term = re.findall('\d+', lbl)
                if not term: 
                    continue
                if 'Over' in lbl:
                    term = ['30']
                new_lbl = term + ['CAD', 'BOND']
                rename_dict [k] = '_'.join(new_lbl)

            df = df[list(rename_dict.keys()) + ['datetime']]
            df = df.rename(columns=rename_dict).set_index('datetime')
            df.to_csv(fname)
            
        return pd.read_csv(fname)

    def get_diff(self, ccy:list):

        df= None
        for pair in ccy:
            if df is None:
                df = getattr(self, pair).copy()
            else:
                df = df.merge(getattr(self, pair), on='datetime', how='outer')
        
        df= df.set_index('datetime')
        cols = [ x for x in df.columns]
        tasks = list(itertools.combinations(cols, 2))
        dfs = []
        for col1, col2 in tasks:
            tmp = df[col1] - df[col2]
            tmp.name = '%s_%s' % (col1, col2)
            dfs.append(tmp)
        return pd.concat(dfs, axis=1)
            

if __name__ == '__main__':
    b =BONDS()
    b.get_diff(['CAD', 'CHF'])
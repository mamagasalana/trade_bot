import requests
import json
import pandas as pd
import os

COMMODITIES = ['gold_am', 'silver', 'palladium_am', 'platinum_am',
               'gold_pm', 'palladium_pm', 'platinum_pm']

class LBMA:
    def __init__(self):
        self.url = 'https://prices.lbma.org.uk/json/{commodity}.json'
        self.folder = 'files/lbma'
        os.makedirs(self.folder, exist_ok=True)

    def get(self, commodity):
        url = self.url.format(commodity=commodity)
        s = requests.Session()
        ret = s.get(url).json()
        return ret
    
    def get_all(self) -> pd.DataFrame:
        f =os.path.join(self.folder, 'all.csv')

        if not os.path.exists(f):
            out = {}
            for c in COMMODITIES:
                js = self.get(c)
                for o in js:
                    dt = o['d']
                    if not dt in out:
                        out[dt] = {}
                    
                    out[dt][c] = o['v'][0]
            pd.DataFrame.from_dict(out, orient='index').sort_index().to_csv(f)

        return pd.read_csv(f, index_col=0)
    
if __name__ == '__main__':
    a = LBMA()
    a.get_all()
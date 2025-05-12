from finagg.fred.api import CategorySeries, SeriesObservations,SeriesCategories,Tags
import datetime
import pandas as pd
import os
from src.csv.lbma import LBMA
import logging
import numpy as np

FRED_MAP = {
 'AUD' : 32269,
 'CHF' : 32279,
 'NZD' : 32338,
 'EUR' : 32947,
 'JPY' : 32281,
 'CAD' : 32268,
 'USD' : 32267,
 'GBP' : 32280,
# 'EUR' : 32273, #germany
# 'EUR' : 32272, #france
# 'EUR' : 32274, #italy
# 'EUR' : 32277, #spain

 }

CCY_MAP = { 
    'DEXCAUS' : 'USDCAD',
    'DEXSZUS' : 'USDCHF',
    'DEXUSEU' : 'EURUSD',
    'DEXUSAL' : 'AUDUSD',
    'DEXUSNZ' : 'NZDUSD', 
    'DEXJPUS' : 'USDJPY',
    'DEXUSUK' : 'GBPUSD',
}

COMMODITY_MAP = { 
    'DCOILWTICO' : 'OILUSD',
    'DHHNGSP' : 'GASUSD',
    'NASDAQCOM' : 'NDQUSD'
}

LBMA_MAP = {
    'gold' : 'XAUUSD',
    'silver': 'XAGUSD',
    'palladium': 'XPDUSD',
    'platinum' : 'XPTUSD' 
}

class FRED:
    def __init__(self, end_year=2024, start_year=2009):
        self.start_year = start_year
        self.end_year= end_year
        self.usd_ccy_list = None
        if not os.path.isdir(f'files/fred'):
            os.makedirs(f'files/fred' , exist_ok=True)
    
    @property
    def filename(self):
        return f'files/fred/{self.ccy}.csv'
    

    def get_currency_to_usd(self, currency):
        """
        Returns a Series representing the USD value of one unit of `currency`.
        If the pair is quoted as 'CURUSD', it's direct.
        If it's quoted as 'USDCUR', we return the reciprocal.
        """
        if self.usd_ccy_list is None:
            self.usd_ccy_list = self.get_all()
        # If the currency is already USD, then 1 USD = 1.
        if currency == 'USD':
            return 1
        # Check for direct pair like 'AUDUSD'
        direct = currency + 'USD'
        if direct in self.usd_ccy_list.columns:
            return self.usd_ccy_list[direct]
        # Otherwise, check for inverse pair like 'USDAUD'
        inverse = 'USD' + currency
        if inverse in self.usd_ccy_list.columns:
            return 1 / self.usd_ccy_list[inverse]
        # If neither is available, raise an error
        raise KeyError(f"Conversion for currency {currency} not found in DataFrame columns.")

    def get_pairs(self, pairs: list):
        """
        get pairs from fred, manually generate exotic pairs 
        """
        if self.usd_ccy_list is None:
            self.usd_ccy_list = self.get_all()
            
        for pair in pairs:
            # If the pair is already present, skip it.
            if pair in self.usd_ccy_list.columns:
                continue

            # Extract base and quote currencies (first 3 letters are base, last 3 are quote)
            base, quote = pair[:3], pair[3:]
            
            # Get USD conversion for base and quote
            base_to_usd = self.get_currency_to_usd(base)
            quote_to_usd = self.get_currency_to_usd(quote)
            exotic_rate = base_to_usd / quote_to_usd

            # Add the new exotic pair column to your DataFrame
            self.usd_ccy_list[pair] = exotic_rate
            logging.debug(f"Generated exotic pair {pair}.")
        return self.usd_ccy_list[pairs].dropna()
    
    def get_all(self) -> pd.DataFrame:
        """Customize your pairs here

        Returns:
            pd.DataFrame: return combined pairs
        """
        df1 = self.get_ccy()
        df2 = self.get_commodity()
        df3 = LBMA().get_all()
        df3 = df3.drop(columns=[col for col in df3.columns if '_pm' in col])
        df3.columns = [LBMA_MAP[col.split('_')[0]] for col in df3.columns]
        
        df = df1.merge(df2, how='outer', right_index=True, left_index=True)
        df = df.merge(df3, how='outer', right_index=True, left_index=True )
        # df.dropna(inplace=True)
        df[df < 0] = np.nan
        return df.dropna()



    def get_commodity(self):
        self.ccy = 'commodity'
        self.start_year = 1970
        # web cache only last for 1 week , is it good to store locally to avoid spamming fred
        if os.path.exists(self.filename):
            df= pd.read_csv(self.filename).pivot(index='datetime', columns='event', values='actual' )
            return df.rename(columns=COMMODITY_MAP)
        df= self.get_event_by_ids(COMMODITY_MAP.keys())
        return df.pivot(index='datetime', columns='event', values='actual').rename(columns=COMMODITY_MAP)   
    
    def get_ccy(self):
        self.ccy = 'ccy'
        self.start_year = 1970
        # web cache only last for 1 week , is it good to store locally to avoid spamming fred
        if os.path.exists(self.filename):
            df= pd.read_csv(self.filename).pivot(index='datetime', columns='event', values='actual' )
            return df.rename(columns=CCY_MAP)
        df= self.get_event_by_ids(CCY_MAP.keys())
        return df.pivot(index='datetime', columns='event', values='actual' ).rename(columns=CCY_MAP)    

    def get_tag(self, tag, _list=False):
        self.ccy = tag
        df =Tags.series.get(tag_names=tag, paginate=True)  

        df['last_updated'] = df.last_updated.apply(lambda x : datetime.datetime.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))
        df['o1'] = pd.to_datetime(df.observation_start)
        df['o2'] = pd.to_datetime(df.observation_end)

        condition1 = [
                     f"(o1.dt.year < {self.start_year})",
                      f"(o2.dt.year >= {self.end_year})"]
        condition2 = [
                       f"(o1.dt.year < {self.start_year})", 
                       f"(o2.dt.year >= {self.end_year-1})"]
        
        condition_groups = [condition1, condition2]
        conditions  = f" or ".join([f"({' and '.join(group)})" for group in condition_groups])

        if _list:
            return df.query(conditions)
        else:
            # web cache only last for 1 week , is it good to store locally to avoid spamming fred
            if os.path.exists(self.filename):
                return pd.read_csv(self.filename)
            return self.get_event_by_ids(df.query(conditions).id.unique())
    
    def get(self, ccy, _list=False):
        self.ccy = ccy
        category_id = FRED_MAP[self.ccy]
        df = CategorySeries().get(category_id, paginate=True)
        df['last_updated'] = df.last_updated.apply(lambda x : datetime.datetime.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))
        df['o1'] = pd.to_datetime(df.observation_start)
        df['o2'] = pd.to_datetime(df.observation_end)

        condition1 = ["(frequency_short in ['Q', 'M'])",
                     f"(o1.dt.year < {self.start_year})",
                      f"(o2.dt.year >= {self.end_year})"]
        condition2 = ["(frequency_short == 'A')" ,
                       f"(o1.dt.year < {self.start_year})", 
                       f"(o2.dt.year >= {self.end_year-1})"]
        
        condition_groups = [condition1, condition2]
        conditions  = f" or ".join([f"({' and '.join(group)})" for group in condition_groups])
        
        if _list:
            return df.query(conditions)
        else:
            # web cache only last for 1 week , is it good to store locally to avoid spamming fred
            if os.path.exists(self.filename):
                return pd.read_csv(self.filename)
            return self.get_event_by_ids(df.query(conditions).id.unique())
        
    def get_event_by_ids(self, eventids):
        so = SeriesObservations()
        sc = SeriesCategories()
        ret= []
        try:
            for eventid in eventids:
                metadata = sc.get(eventid)
                eclass = metadata.iloc[-1]['name']
                df = so.get(eventid )[['series_id', 'value', 'date']]
                df.columns= ['event', 'actual', 'datetime']
                df['currency'] = self.ccy
                df['eclass'] = eclass
                df['datetime'] = pd.to_datetime(df.datetime)
                ret.append(df)
        except:
            print(list(eventids).index(eventid))
            raise
        out =  pd.concat(ret).reset_index(drop=True)
        out.to_csv(self.filename)
        return out

if __name__ =='__main__':
    a = FRED()
    df = a.get_all()
    print('debug')
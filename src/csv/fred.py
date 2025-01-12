from finagg.fred.api import CategorySeries, SeriesObservations,SeriesCategories
import datetime
import pandas as pd
import os

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

class FRED:
    def __init__(self, end_year=2024):
        self.start_year = 2009
        self.end_year= end_year
    
    @property
    def filename(self):
        return f'files/fred/{self.ccy}.csv'

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
                ret.append(df[df.datetime.dt.year >=self.start_year])
        except:
            print(list(eventids).index(eventid))
            raise
        out =  pd.concat(ret).reset_index(drop=True)
        out.to_csv(self.filename)
        return out

if __name__ =='__main__':
    a = FRED()
    a.get('AUD')
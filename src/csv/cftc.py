import pandas as pd
from sodapy import Socrata
from src.csv.cache import CACHE2

my_cache =  CACHE2('cftc.cache')


CCY_MAPPING = {
    'CAD' : '090741',
    'CHF' : '092741',
    'GBP' : '096742',
    'JPY' : '097741',
    'EUR' : '099741',
    'NZD' : '112741',
    'AUD' : '232741',
    'USD' : '098662',
    'OIL' : '067651',
    'XAU' : '088691',
    'XAG' : '084691',
    'GAS' : '0233AG',

}

class CFTC:
# Example authenticated client (needed for non-public datasets):
    def __init__(self):
        self.client = Socrata("publicreporting.cftc.gov", 'Nh9Qta0y4CmwZn0Ml90OikVAa')

    @my_cache
    def get_data(self, ccy, force_reset=False):
        code = CCY_MAPPING[ccy]
        results  = self.client.get("6dca-aqww", where="cftc_contract_market_code = '%s'" % code, limit=3000)
        return pd.DataFrame(results)


    def get_all_data(self, force_reset=False):
        dfs = []
        for ccy in CCY_MAPPING:
            df = self.get_data(ccy, force_reset)
            df['ccy'] = ccy
            dfs.append(df)
        
        ret =  pd.concat(dfs).sort_values(by='report_date_as_yyyy_mm_dd').reset_index(drop=True)

        ret['dt'] = pd.to_datetime(ret.report_date_as_yyyy_mm_dd)
        ret = ret[[x for x in ret.columns if 
                   not x.endswith('_other') and
                   not x.endswith('_old') and
                   not x.endswith('_1') and
                   not x.endswith('_2') and
                   not x.startswith('pct_') and
                   not 'tot_' in x and
                   not x.startswith('change_') and
                   not x in ['noncomm_positions_spread', 'noncomm_positions_spread_1']
                   ]].copy()

        return ret

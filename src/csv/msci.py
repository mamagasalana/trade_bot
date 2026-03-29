import os
from datetime import date

import pandas as pd
import requests
import json
from tqdm import tqdm

class MSCI:
    WORLD_CODE = [705130, 129857, 729749, 136064, 129896, 703755, 702787]
    USA_CODE = [705973, 129858, 729754, 129788, 139133, 703025, 702789]


    def __init__(self):
        self.url = 'https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph'
        self.folder = 'files/msci'
        os.makedirs(self.folder, exist_ok=True)
        self.s = requests.Session()
        self.MSCI_CODES = self.get_all_msci_codes()
        self.CODE_MAP = {}
        self.CODE_MAP_REVERSE = {}

    def get_all_msci_codes(self, raw=False):
        FILE = os.path.join(self.folder, 'MSCI_CODES.json')
        if not os.path.exists(FILE):
            print('Downloading ... ')
            # ytee 29 March 2026: not sure if this works in other case # TODO
            url = "https://www.msci.com/indexes"
            headers = {
                "accept": "text/x-component",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "cache-control": "no-cache",
                "connection": "keep-alive",
                "content-type": "text/plain;charset=UTF-8",
                "host": "www.msci.com",
                "next-action": "aa5adaca031c212dd47cf0ad1af15b5e40374ce7",
                "next-router-state-tree": "%5B%22%22%2C%7B%22children%22%3A%5B%22(main)%22%2C%7B%22children%22%3A%5B%22__PAGE__%3F%7B%5C%22index-page%5C%22%3A%5C%22109%5C%22%7D%22%2C%7B%7D%2C%22%2Findexes%3Findex-page%3D4%22%2C%22refresh%22%5D%7D%5D%2C%22infoBar%22%3A%5B%22__DEFAULT__%22%2C%7B%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D",
                "origin": "https://www.msci.com",
                "pragma": "no-cache",
                "referer": "https://www.msci.com/indexes",
                "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            }

            payload = [
                {
                    "filters": {
                        "index-type": ["Equity"],
                        "variant": ["GRTR"],
                        "currency": ["USD"],
                        "keyword": "$undefined",
                    },
                    "sortBy": "featuredAsc",
                    "page": [0, 3715],
                    "performanceVariant": "GRTR",
                    "performanceCurrency": "USD",
                    "calcDate": "2026-03-27",
                },
                None,
                "$undefined",
            ]

            response = requests.post(url, headers=headers, json=payload)
            data = json.loads(response.text.split('\n')[1][2:])
            with open(FILE, 'w') as ofile:
                json.dump(data, ofile)
        data = json.load(open(FILE, 'r'))
        if raw:
            return data
        self.CODE_MAP =  {x['indexCode'] : x['indexName'] for x in data['data']}
        self.CODE_MAP_REVERSE = {v:k for k,v in self.CODE_MAP.items()}
        selected_keys = ['indexCode','indexName','indexInceptionDate','indexType','sizes','markets','currencies','taxonomyGroups','taxonomyCategories','taxonomyRegion','region','country']
        truncated_dict = [{k: item.get(k) for k in selected_keys} for item in data['data']]
        df = pd.DataFrame(truncated_dict)
        return df

    def _build_params(self, code):
        return {
            'currency_symbol': 'USD',
            'index_variant': 'STRD',
            'start_date': '20010101',
            'end_date': date.today().strftime('%Y%m%d'),
            'data_frequency': 'DAILY',
            'index_codes': code,
        }

    def _to_frame(self, payload, code):
        rows = payload.get('indexes', {}).get('INDEX_LEVELS', [])
        if not rows:
            return pd.DataFrame(columns=[code])

        df = pd.DataFrame(rows)
        date_col = 'calc_date' if 'calc_date' in df.columns else 'date'
        value_col = 'level_eod' if 'level_eod' in df.columns else 'level'

        df = df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: code})
        df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
        df[code] = pd.to_numeric(df[code], errors='coerce')
        return df.set_index('date').sort_index()

    def get(self, code, raw=False) -> pd.DataFrame:
        params = self._build_params(code=code)
        
        ret = self.s.get(self.url, params=params, timeout=30).json()
        if raw:
            return ret
        return self._to_frame(ret, code)

    def get_all(self, country='US', force=False) -> pd.DataFrame:
        f = os.path.join(self.folder, f'{country}.parquet')
        
        if country =='US':
            SELECTED_MSCI_CODES = self.USA_CODE
        elif country =='World':
            SELECTED_MSCI_CODES = self.WORLD_CODE
        else:
            raise LookupError("Country not found")
        
        if not os.path.exists(f) or force:
            dfs = []
            for code in tqdm(SELECTED_MSCI_CODES):
                df = self.get(code)
                dfs.append(df)

            if dfs:
                out = pd.concat(dfs, axis=1).sort_index()
            else:
                out = pd.DataFrame()

            out.to_parquet(f)

        df =  pd.read_parquet(f)
        df.columns = df.columns.astype(str).map(self.CODE_MAP).str.findall('MSCI \S+ (.\D*) Index').str[0]
        return df


if __name__ == '__main__':
    app = MSCI()
    app.get_all(force=True)

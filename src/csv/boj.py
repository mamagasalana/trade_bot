import requests
from lxml import html
import re
import os
import json
import time
from tqdm import tqdm
from urllib.parse import urljoin
import pandas as pd


class BOJ:
    def __init__(self):
        self.session = requests.Session()
        self.validation= []
        self.metadata_cache = None
        if not os.path.isdir( 'files/boj'):
            os.makedirs('files/boj')
            
    @property
    def header_categories(self):
        return {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Host': 'www.stat-search.boj.or.jp',
            'Pragma': 'no-cache',
            'Sec-CH-UA': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
        }

    @property
    def headers_list_of_series(self):
        return {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Host': 'www.stat-search.boj.or.jp',
            'Origin': 'https://www.stat-search.boj.or.jp',
            'Pragma': 'no-cache',
            'Sec-CH-UA': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        }
    @property
    def header_series_metadata(self):
        return {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Host': 'www.stat-search.boj.or.jp',
            'Origin': 'https://www.stat-search.boj.or.jp',
            'Pragma': 'no-cache',
            'Referer': 'https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2?cgi=$nme_s050_en',
            'Sec-CH-UA': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
        }

    
    def get_categories(self):
        url = 'https://www.stat-search.boj.or.jp/index_en.html'
        r = self.session.get(url, headers=self.header_categories)
        tree = html.fromstring(r.content)        
        x1 = '//ul[@class="toukeimenu parentMenuList"]//a[@class="childMenuItem"]/@href'
        urls = tree.xpath(x1)
        return [ re.findall("lstSelection=(.*)", x)[0] for x in urls]

    def get_categories_cache(self):
        FILES = 'files/boj/categories.json'
        if os.path.exists(FILES):
            with open('files/boj/categories.json', 'r') as ifile:
                js = json.load(ifile)
        else:
            js = {}
        return js
    
    def set_categories_cache(self, js):
        FILES = 'files/boj/categories.json'
        with open(FILES, 'w') as ofile:
            json.dump(js, ofile)

    def get_metadata_cache(self):
        FILES = 'files/boj/series_metadata.csv'
        if self.metadata_cache  is not None:
            return self.metadata_cache
        
        if os.path.exists(FILES):
            df = pd.read_csv(FILES)
        else:
            headers = ['Series code', 'Name of time-series', 'Statistical category',
                    'Frequency', 'Conversion method', 'Last update', 'Unit',
                    'Start of the time-series', 'End of the time-series', 'Notes']

            # Create an empty DataFrame with these headers
            df = pd.DataFrame(columns=headers)
        self.metadata_cache = df
        return df

    def set_metadata_cache(self, js):
        FILES = 'files/boj/series_metadata.csv'
        df1 = pd.DataFrame(js)
        df2 = self.get_metadata_cache()
        df = pd.concat([df2,df1]).drop_duplicates()
        df.to_csv(FILES, index=False)
        self.metadata_cache = df

    def get_list_of_series(self, code, echelon=''):
        if not echelon:
            js = self.get_categories_cache()
            if code in js:
                return js[code]

        out = []
        url = 'https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2'
        
        echelon_count = 0
        if echelon:
            echelon_count = echelon.count(',') + 1
        cgi= f"$nme_a01{echelon_count}_en"

        assert (cgi , code, echelon ) not in self.validation, f"already run? {(cgi , code, echelon )}"
        self.validation.append((cgi , code, echelon ) )
        form_data = {
            'cgi': cgi,
            'obj_name': code,
            'lstCodelist': echelon
        }
        time.sleep(1)
        r = self.session.post(url, headers=self.headers_list_of_series, data=form_data)
        try:
            ret= r.json()
            if ret['isTerminate']:
                return [ itm['code'] for itm in ret['items']]
            else:
                for itm in ret['items']:
                    echelon2 = itm['echelon']
                    echelon2 = ', '.join(map(str, echelon2))
                    ret2 = self.get_list_of_series(code, echelon2)
                    out.extend(ret2)

        except:
            time.sleep(3)
            self.validation.remove((cgi , code, echelon ))
            return self.get_list_of_series(code, echelon)

        if not echelon:
            js[code] = out
            self.set_categories_cache(js)
            
        return out
    
    def get_metadata_main(self, code):
        df = self.get_metadata_cache()
        js = self.get_categories_cache()

        series_id_list_pre = js.get(code)
        series_id_list_pre = [f"{code}'{x}" for x in series_id_list_pre ]
        unique_series_codes = set(df['Series code'].unique())

        # Filter list based on unique series codes
        series_id_list = [x for x in series_id_list_pre if x not in unique_series_codes]

        if not series_id_list:
            return
        
        max_iter = (len(series_id_list)-1 )// 240 +1

        for idx in range(max_iter):
            series_ids = series_id_list[idx*240: (idx+1)*240]
            assert series_ids, "series id is empty"
            tmp = self.get_metadata(series_ids)
            
            self.set_metadata_cache(tmp)

    def get_metadata(self, txtfield):
        url = 'https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2'
        form_data = {
            'cgi': '$nme_r000_en',
            'txtYyyyFrom': '',  # Assuming empty value is intentional
            'txtYyyyTo': '',  # Assuming empty value is intentional
            'cmbFREQ': 'NONE',
            'cmbFREQ_OPTION': 'NONE',
            'lstCode': txtfield
        }
        x1 = '//a[contains(text(), "Information on")]/@href'
        x2 = '//table'
        r = self.session.post(url, headers=self.header_series_metadata, data=form_data)
        tree = html.fromstring(r.content)
        urls = [urljoin(url, x) for x in tree.xpath(x1)]
        out = []
        for url2 in urls:
            r2 = self.session.get(url2, headers=self.header_categories)
            tree = html.fromstring(r2.content)
            for body in tree.xpath(x2):
                headers =body.xpath('.//th')
                data  = body.xpath('.//td')
                out.append( {h.text_content().strip(): d.text_content().strip() for h, d in zip(headers, data)})

        assert out, "out is empty?"
        return out
    
if __name__ =='__main__':
    a = BOJ()
    categories = a.get_categories()
    
    
    for c in tqdm(categories):
        meta =a.get_metadata_main(c)
    #     series = a.get_list_of_series(c)
        
    
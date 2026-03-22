import json
from pathlib import Path
import pandas as pd
from lxml import html
from patchright.async_api import async_playwright
import nest_asyncio
nest_asyncio.apply()
import asyncio
import re

USER_DATA_DIR = "jupyter/mm_chrome_profile"
PATCHRIGHT_LAUNCH_ARGS = {
    "channel": "chrome",
    "headless": False,
    "no_viewport": True,
}

class MM:
    def __init__(self):
        self.loading= True

    def get_json(self, url):
        self.loading=True
        self.url = url
        DATA_ID = re.findall(r'charts/(\d+)', url,)[0]
        self.url_data=  f"charts/data/{DATA_ID}"
        self.CHART_JSON_FILE = Path(f"files/macromicro/{DATA_ID}.json")
        
        if not self.CHART_JSON_FILE.exists():
            asyncio.run(self.open_chart_page())

        js= json.loads(self.CHART_JSON_FILE.read_bytes())
        names = [x['name_en'] for x in js['data']['c:%s' % DATA_ID]['info']['chart_config']['seriesConfigs']]
        series_list = js['data']['c:%s' % DATA_ID]['series']

        dfs = []
        for series_name, s in zip(names, series_list):
            df = pd.DataFrame(s, columns=["date", series_name])
            df["date"] = pd.to_datetime(df["date"])
            df[series_name] = pd.to_numeric(df[series_name])
            df = df.set_index('date')
            dfs.append(df)
                
        all_df = pd.concat(dfs,axis=1).sort_index()
        return all_df

    
    async def open_chart_page(self, wait_ms: int = 30000):
        # get cache first
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=USER_DATA_DIR,
                **PATCHRIGHT_LAUNCH_ARGS,
            )
            page = context.pages[0] if context.pages else await context.new_page()

            async with page.expect_response(
                lambda r: self.url_data in r.url,
                timeout=wait_ms
            ) as resp_info:
                await page.goto(self.url, wait_until="domcontentloaded")

            response = await resp_info.value
            chart_json = await response.json()

            self.CHART_JSON_FILE.write_text(
                json.dumps(chart_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )




if __name__ == '__main__':
    app = MM()
    app.get_json("https://sc.macromicro.me/charts/35559/china-credit-impulse-index")
    # print('debug')
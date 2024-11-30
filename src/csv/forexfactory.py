
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:25:35 2022

@author: Butterfly2
"""

# https://trends.builtwith.com/websitelist/DataDome
# Test websites from here
from chromedriver import CHROME
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
import os
import random
import time
import requests
import pandas as pd
import datetime
import re
from lxml import html
import json
# https://stackoverflow.com/questions/55582136/how-to-set-proxy-with-authentication-in-selenium-chromedriver-python    
from src.maps.config import *

class ForexFactoryData:
    def __init__(self):
        self.c = CHROME()
        self.driver = self.c.launch_driver(sleep=False)
    
    def main(self, startdt = None):
        
        excl = []
        if os.path.exists(FILES):
            df_exist = pd.read_csv(FILES)
            excl = df_exist.date.unique().tolist()

        if startdt is not None:
            dt = startdt
        else:
            dt = datetime.datetime(2014,1,1)
        dflist = []

        if dt > datetime.datetime.now():
            print("Early exit")
            return dt
        
        try:
            while dt <= datetime.datetime.now():
                if dt.strftime('%Y-%m-%d') not in excl:
                    dflist.append(self.get_historical(dt))
                dt += datetime.timedelta(days=1)
        except Exception as e:
            print('############   Fail  ############')
            # print(e)
            print(dt)

        if dflist:
            df2 = pd.concat(dflist, ignore_index=True)
        else:
            df2 = pd.DataFrame()

        if not df2.empty:
            df2['datetime'] = pd.to_datetime(df2['date']+' ' +df2['time'], format='%Y-%m-%d %I:%M%p', errors='coerce')
            if os.path.exists(FILES):
                df_exist = pd.read_csv(FILES)
                dffinal = pd.concat([df_exist, df2], axis=0)
                dffinal.to_csv(FILES, index=False)
            else:
                df2.to_csv(FILES, index=False)
        
        if dt != startdt:
            print("Resuming ...")
            self.c.launch_driver()
            self.main(dt)
        print(dt, startdt)

    def get_historical(self, startdate):
        x1 = '//table[@class="calendar__table"]'
        if os.name == 'nt':
            dt = startdate.strftime('%b%#d.%Y').lower()
        else:
            dt = startdate.strftime('%b%-d.%Y').lower()
        self.driver.get(f'https://www.forexfactory.com/calendar?day={dt}')
        # e1 = WebDriverWait(self.driver, 15).until(EC.visibility_of_element_located((By.XPATH, x1)))
        if os.name == 'nt':
            dt2 = startdate.strftime('%b %#d')
        else:
            dt2 = startdate.strftime('%b %-d')
        x2 = f'//span[text()="{dt2}"]'
        WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.XPATH, x2)))
        e1 = WebDriverWait(self.driver, 15).until(EC.visibility_of_element_located((By.XPATH, x1)))
        out = []
        for tr in e1.find_elements(By.XPATH, './/tr[@data-event-id]'):
            out2=  {'date': startdate.strftime('%Y-%m-%d'), 'eventid': tr.get_attribute('data-event-id')}

            for attr in ['time', 'event', 'currency', 'actual', 'forecast', 'previous']:
                x1 = f'.//td[contains(@class, "calendar__{attr}")]'
                out2[attr]= tr.find_element(By.XPATH, x1).text

            x1 = './/td[contains(@class, "calendar__impact")]/span'
            out2['impact']= tr.find_element(By.XPATH, x1).get_attribute('title')
            out.append(out2)

        return pd.DataFrame(out)

    def get_eventids(self):
        URL = 'https://www.forexfactory.com/calendar/details/1-{id}'

        def get_url(eventid):
            self.driver.get('about:blank')
            self.driver.get(URL.format(id=eventid))
            WebDriverWait(self.driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            json_pre_tag = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "pre")))
            # Extract the JSON text
            js = json.loads(json_pre_tag.text)
            return js['data']['specs']

        out = {}
        if os.path.exists(FILES_EVENTID):
            out = json.load(open(FILES_EVENTID))
        
        df = pd.read_csv('files/forexfactory/forexfactory_calendar.csv')
        eventids = df.eventid.unique().tolist()
        for eventid in eventids:
            if str(eventid) in out:
                continue

            for _ in range(3):
                try:
                    ret = get_url(eventid)
                    break
                except:
                    with open(FILES_EVENTID, 'w') as ifile:
                        ifile.write(json.dumps(out))
                    self.c.launch_driver()

            out[eventid] = ret

        
        with open(FILES_EVENTID, 'w') as ifile:
            ifile.write(json.dumps(out))
        print('debug')

if __name__ == '__main__':
    app = ForexFactoryData()
    dt= datetime.datetime(2024,8,1)
    # app.main(dt)
    app.get_eventids()

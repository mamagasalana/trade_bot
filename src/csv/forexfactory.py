
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:25:35 2022

@author: Butterfly2
"""

# https://trends.builtwith.com/websitelist/DataDome
# Test websites from here


import undetected_chromedriver as uc

# Initializing driver 

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import subprocess



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

FILES = r'files/forexfactory/forexfactory_calendar.csv'

class ForexFactoryData:
    def __init__(self):
        self.driver = self.launch_driver()

    def launch_driver(self, folder=None):
        subprocess.run(["pkill", "chrome"])
        # prevent anti-bot guide
        # https://piprogramming.org/articles/How-to-make-Selenium-undetectable-and-stealth--7-Ways-to-hide-your-Bot-Automation-from-Detection-0000000017.html
        # https://www.zenrows.com/blog/selenium-avoid-bot-detection#alternatives-to-base-selenium

        # # selenium-wire proxy settings
        capa = DesiredCapabilities.CHROME
        capa["pageLoadStrategy"] = "none"
        # capa["unexpectedAlertBehaviour"] = "accept"
        options = webdriver.ChromeOptions()

        chromeprofile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hwqnghwgflanliuvfgqe")
        if not os.path.exists(chromeprofile):
            os.makedirs(chromeprofile)
        
        options.add_argument('--user-data-dir=' +chromeprofile)
            
        if folder:
            prefs = {   'download.default_directory' : folder,
                        "download.prompt_for_download": False,
                        "download.directory_upgrade": True,
                        "plugins.always_open_pdf_externally": True}
            options.add_experimental_option('prefs', prefs)
        
        driver = uc.Chrome(options=options, desired_capabilities=capa) 
        driver.maximize_window()
        return driver
    
    def main(self, startdt = None):

        if startdt is not None:
            dt = startdt
        else:
            dt = datetime.datetime(2009,1,1)
        dflist = []

        if dt > datetime.datetime.now():
            return dt
        
        try:
            while dt <= datetime.datetime.now():
                dflist.append(self.get_historical(dt))
                dt += datetime.timedelta(days=1)
        except Exception as e:
            print('############   Fail  ############')
            # print(e)
            print(dt)

        df2 = pd.concat(dflist, ignore_index=True)
        df2['datetime'] = pd.to_datetime(df2['date']+' ' +df2['time'], format='%Y-%m-%d %I:%M%p', errors='coerce')
        if os.path.exists(FILES):
            df_exist = pd.read_csv(FILES)
            dffinal = pd.concat([df_exist, df2], axis=0)
            dffinal.to_csv(FILES, index=False)
        else:
            df2.to_csv(FILES, index=False)
        print('Done')
        return dt
    
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
            out2=  {'date': startdate.strftime('%Y-%m-%d')}

            for attr in ['time', 'event', 'currency', 'actual', 'forecast', 'previous']:
                x1 = f'.//td[contains(@class, "calendar__{attr}")]'
                out2[attr]= tr.find_element(By.XPATH, x1).text

            x1 = './/td[contains(@class, "calendar__impact")]/span'
            out2['impact']= tr.find_element(By.XPATH, x1).get_attribute('title')
            out.append(out2)

        return pd.DataFrame(out)

if __name__ == '__main__':
    app = ForexFactoryData()

    dt= datetime.datetime(2009,1,1)

    while True:
        dt2 = app.main(dt)

        if dt2 ==dt:
            break
        
        app.driver = app.launch_driver()
        dt =dt2

    print('Done')
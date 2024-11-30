import requests
import subprocess
import os

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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from seleniumbase import Driver
import time


class CHROME:
    def __init__(self):
        pass

    def download(self, url):
        FILE = url.split('/')[-1].replace('.zip', '')
        download_path = "/tmp/test.zip"
        extract_path1 = '/tmp/test/'
        extract_path = f"/tmp/test/{FILE}/{FILE.split('-')[0]}"
        install_path = "/usr/local/bin/"

        # Step 1: Download the file using wget
        download_command = f"wget {url} -O {download_path}"
        subprocess.run(download_command, shell=True, check=True)

        # Step 2: Unzip the file
        unzip_command = f"unzip {download_path} -d {extract_path1}"
        subprocess.run(unzip_command, shell=True, check=True)

        # Step 3: Move the necessary files to /usr/local/bin/
        move_command = f"sudo mv {extract_path} {install_path}"
        subprocess.run(move_command, shell=True, check=True)

        # Step 4: Clean up
        os.remove(download_path)
        os.system(f"rm -rf {extract_path1}")

    def update(self, version):
        js = requests.get('https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json').json()

        chrome = [x for x in js['versions'] if x['version'].startswith(str(version))][-1]

        lx = [x for x in chrome['downloads']['chrome'] if x['platform'] == 'linux64'][0]
        lx2 = [x for x in chrome['downloads']['chromedriver'] if x['platform'] == 'linux64'][0]

        # self.download(lx['url'])
        self.download(lx2['url'])

    def launch_driver(self, folder=None, sleep=True):
        subprocess.run(["pkill", "chrome"])
        # prevent anti-bot guide
        # https://piprogramming.org/articles/How-to-make-Selenium-undetectable-and-stealth--7-Ways-to-hide-your-Bot-Automation-from-Detection-0000000017.html
        # https://www.zenrows.com/blog/selenium-avoid-bot-detection#alternatives-to-base-selenium
        if sleep:
            time.sleep(60)
        # # selenium-wire proxy settings
        capa = DesiredCapabilities.CHROME
        capa["pageLoadStrategy"] = "none"
        # capa["unexpectedAlertBehaviour"] = "accept"
        # options = webdriver.ChromeOptions()

        chromeprofile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hwqnghwgflanliuvfgqe")
        if not os.path.exists(chromeprofile):
            os.makedirs(chromeprofile)
        
        options = uc.ChromeOptions()
        options.add_argument("--headless")  # headless
        options.add_argument("--disable-gpu")  # Disable GPU
        driver = Driver(uc=True,headless=True)

        # driver = uc.Chrome(driver_executable_path='/usr/local/bin/chromedriver',
        #     user_data_dir=chromeprofile,options=options, debug=True)
        # driver.maximize_window()
        return driver
    
# to test if chromedriver works, paste this into command line    
"""
chromedriver --port=9515

curl -X POST http://localhost:9515/session -H "Content-Type: application/json" -d '{
  "capabilities": {
    "alwaysMatch": {
      "goog:chromeOptions": {
        "args": ["--disable-gpu", "--no-sandbox", "--start-maximized"]
      }
    }
  }
}'
"""
if __name__ == '__main__':
    c = CHROME()
    c.update(131)

# -*- coding: utf-8 -*-

# ! pip install selenium
# ! pip install fake_useragent
# ! pip install webdriver_manager

# !apt update
# !apt install chromium-chromedriver

import os
import re
import pandas as pd
import pickle
import collections
from collections import defaultdict
import numpy as np
import math
from ast import literal_eval
from time import gmtime, strftime
import re
import time
from bs4 import BeautifulSoup as BeautifulSoup
from tqdm.auto import tqdm

# Scrapping
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

# Error Handling
import socket
import urllib3
import urllib.request
from urllib.request import urlopen
from urllib.parse import quote_plus
from urllib.request import urlretrieve
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, ElementClickInterceptedException
import warnings
warnings.filterwarnings('ignore')

from tqdm.auto import tqdm


def parsing(driver):
  html = driver.page_source
  soup = BeautifulSoup(html, "html.parser")
  return soup

category_dict = {'스킨케어': 'https://shopping.naver.com/beauty/category?menu=10003291&sort=REVIEW',
                 '메이크업': 'https://shopping.naver.com/beauty/category?menu=10003329&sort=REVIEW',
                 '바디케어': 'https://shopping.naver.com/beauty/category?menu=10003353&sort=REVIEW',
                 '맨즈케어': 'https://shopping.naver.com/beauty/category?menu=10003399&sort=REVIEW'}

# set webdriver option
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

socket.setdefaulttimeout(30)

url_dict = {}

for category_name, category_url in category_dict.items():
    driver = webdriver.Chrome(options=options)
    driver.get(category_url)
    soup = parsing(driver)

    prd_url_lst = []
    begin, end = 0, 700
    while len(prd_url_lst) < 100:
        prd_lst = soup.find_all("li", {"class":"MK7X28zO7G"})
        for prd in prd_lst:
            match = re.search(r"stores\/\d+\/products\/\d+", str(prd.find_all("a", {"class":"_1Q9SQDKhCz"})[0]))
            store_id = match.group().split('/')[1]
            product_id = match.group().split('/')[3]
            prd_url = "https://shopping.naver.com/beauty/stores/" + store_id + "/products/" + product_id
            prd_url_lst.append(prd_url)
            prd_url_lst = list(set(prd_url_lst))

        driver.execute_script(f"window.scrollTo({begin}, {end})")
        time.sleep(1)
        begin = end
        end += 700
        soup = parsing(driver)

    url_dict[category_name] = prd_url_lst

def scrap_review_each_page(review_each_page):
    review_contents = []
    ratings = [int(rating.text) for rating in [profile.find("em", {"class":"_15NU42F3kT"})
                                              for profile in soup.find_all("div", {"class":"_1rZLm75kLm"})]]
    highlights = [highlight.text for highlight in [div.find("em", {"class":"_2_otgorpaI"}) for div in soup.find_all("div", {"class":"YEtwtZFLDz"})]]
    for i in range(len(review_each_page.find_all("div", {"class":"YEtwtZFLDz"}))):
        review_contents.append([review_text.text for review_text in [review_each_page.find_all("span", {"class":"_3QDEeS6NLn"})
                                                                     for review_each_page in review_each_page.find_all("div", {"class":"YEtwtZFLDz"})][i]])
    return ratings, highlights, review_contents

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def scrap_product_review(prd_url):
    driver = webdriver.Chrome(options=options)
    driver.get(prd_url)
    soup = parsing(driver)
    actions = ActionChains(driver)

    topic_names = [i.text for i in [topic.find_all('a', {'class':'_1ninPXcmCV'}) for topic in soup.find_all('ul', {'id':'topic_ul'})][0]][1:]
    if len(topic_names) < 2:
      return None

    prd_review_dict = {}
    for i in range(len(topic_names)):
        topic_ul = driver.find_element_by_css_selector('#topic_ul')
        actions.move_to_element(topic_ul).perform()
        time.sleep(0.5)
        driver.find_element_by_css_selector(f'#topic_ul > li:nth-child({i+2}) > a').click()
        time.sleep(0.5)
        soup = parsing(driver)
        prd_review_dict[topic_names[i]] = []
        ratings_all, highlights_all, review_contents_all = [], [], []
        ratings, highlights, review_contents = scrap_review_each_page(soup)
        ratings_all.append(ratings); highlights_all.append(highlights); review_contents_all.append(review_contents)
        while True:
            try:
                driver.find_element_by_css_selector('#REVIEW > div > div.hmdMeuNPAt > div > div._1QyrsagqZm._2w8VqYht7m > a._3togxG55ie._2_kozYIF0B').click()
                time.sleep(0.5)
                soup = parsing(driver)
                ratings, highlights, review_contents = scrap_review_each_page(soup)
                ratings_all.append(ratings); highlights_all.append(highlights); review_contents_all.append(review_contents)
            except ElementNotInteractableException:
                break
        prd_review_dict[topic_names[i]] = [flatten(ratings_all), flatten(highlights_all), flatten(review_contents_all)]

    driver.close()
    return prd_review_dict

skincare_prd = {}
for prd_url in tqdm(url_dict['스킨케어']):
  prd_review_dict = scrap_product_review(prd_url)
  skincare_prd[prd_url] = prd_review_dict
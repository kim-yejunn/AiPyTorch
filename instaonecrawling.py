from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from urllib.request import urlretrieve
import os


url = "https://www.instagram.com/p/C6tmVaCyedc/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=="
name = "image01"

driver = webdriver.Chrome()
driver.get(url)
time.sleep(2)
image = driver.find_element(By.CSS_SELECTOR, "img[class*='x5yr21d']").get_attribute('src')
urlretrieve(image, os.path.join("insta", name + '.jpg'))
print("Done!")
driver.close()
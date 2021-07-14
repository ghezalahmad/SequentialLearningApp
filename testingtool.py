from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os

#os.system('voila --no-browser SequentialLearningApp.ipynb')

driver = webdriver.Firefox()

driver.get("http://localhost:8866/")
elements = driver.find_elements_by_class_name("lm-Widget p-Widget jupyter-widgets widget-upload jupyter-button")
assert len(elements) >0 
for e in elements:
    e.click()
#assert "No results found." not in driver.page_source
#driver.close()
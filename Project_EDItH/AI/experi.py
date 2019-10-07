
'''
import h5py
filename = 's2s.h5'
from keras.models import load_model
m = load_model(filename)
print(m.summary())



from keras.models import model_from_json
json_file = open('s2s.json', 'rb')
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)

print(load_model)


import urllib.request
import urllib
from bs4 import BeautifulSoup as bs
import requests as req

def search_dining(query):
    enc_query = str(urllib.parse.quote(query))
    url = 'http://www.diningcode.com/list.php?query=' + enc_query
    html = req.get(url).text

    soup = bs(html, 'html.parser')
    addrs = [str(soup.select(f'#div_list > li:nth-child({rep}) > a > span:nth-child(5)')[0]).split('</i>') for rep in range(4, 14)]
    simple_addrs = [addrs[addr][0][35:] for addr in range(len(addrs))]
    detail_addrs = [addrs[addr][1][:-7] for addr in range(len(addrs))]
    score = [soup.select(f'#div_list > li:nth-child({rep}) > p > span.point')[0].text for rep in range(4, 14)]

    return score

query = '서울 디저트'
dining = search_dining(query)
print(dining)
'''

from keras.models import load_model
w_path = './s2s.h5'
m = load_model(w_path)

layers = dict([(layer.name, layer.output) for layer in m.layers])
from pprint import pprint as pp
pp(layers)
print(layers['input_23'][0])
print(layers['input_23'][1])
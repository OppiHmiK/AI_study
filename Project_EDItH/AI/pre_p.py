import pandas as pd
from pprint import pprint

def read_data(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        data = [line.split(':') for line in f.read().splitlines()]
    return data

fpath = 'kakao_dialog.txt'
text_data = read_data(fpath)

sen_lis = []
for text in text_data:
    sen_dict = {'user' : text[0], 'sentence' : text[1]}
    sen_lis.append(sen_dict)

import pickle
with open('kakao_dialog.pickle', 'wb') as f:
    pickle.dump(sen_lis, f)

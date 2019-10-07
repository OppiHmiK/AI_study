from flickrapi import FlickrAPI
from pprint import pprint
from urllib.request import urlretrieve
import os

path = '../Datas/cat_and_dog'
if not os.path.isdir(path):
    os.mkdir(path)
os.chdir(path)


def flickr(keyword):
    # keyword format : '"keyword"' or "'keyword'"
    KEY = 'c13f5b6c16629f36a0168366fbf669a6'
    PW = '4f2b5971cf2e1fd5'
    flickr = FlickrAPI(KEY, PW, format='parsed-json')
    result = flickr.photos.search(text=keyword, per_page='500', sort='relevance')
    photos = result['photos']['photo']
    return photos

def flickr_url(photo, size =''):
    url = 'http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg'
    if size:
        size = '_'+size
    return url.format(size=size, **photo)

cats = flickr('"cat"')
for rep in range(0, len(cats)):
    url = flickr_url(cats[rep])
    fname = '0_cats' + str(rep+1) + '.png'
    urlretrieve(url, fname)

dogs = flickr('"dog"')
for rep in range(0, len(dogs)):
    url = flickr_url(dogs[rep])
    fname = '1_dogs' + str(rep+1) + '.png'
    urlretrieve(url, fname)

print('File saved!')


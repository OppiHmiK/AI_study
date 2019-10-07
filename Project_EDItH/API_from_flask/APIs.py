# -*- coding : utf-8 -*-

from bs4 import BeautifulSoup as bs
import requests as req
from urllib.request import urlopen, Request
from flask import Flask, request, jsonify
from apiclient.discovery import build
import pandas as pd
import random
import time as t
import urllib
import json
import re
import sys

app = Flask(__name__)
naver_key = 'Xbbwo5vSD6gCq_5Fa2P8'
naver_pw = 'hHisHjqN_M'
google_key = 'AIzaSyBYrkhvhfXjdR7CA9EXWKqp5q5rLprMfNs'


@app.route('/weather', methods=['GET', 'POST'])
def weather():
    body = request.get_json()
    loc = body['action']['params']['sys_location']

    if 'sys_location1' in body['action']['params']:
        sub_loc = body['action']['params']['sys_location1']
        enc_loc = str(urllib.parse.quote(loc + sub_loc + '+날씨'))

    else:
        enc_loc = str(urllib.parse.quote(loc + '+날씨'))

    url = 'https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query='+enc_loc
    html = req.get(url).text
    soup = bs(html, 'lxml')

    am_rain_pct = soup.find('li', class_='date_info today').find('span', class_='point_time morning').find('span', class_='rain_rate').text
    pm_rain_pct = soup.find('li', class_='date_info today').find('span', class_='point_time afternoon').find('span', class_='rain_rate').text

    temperature = soup.find('p', class_='info_temperature').find('span', class_='todaytemp').text
    weather1, weather2 = soup.find('ul', class_='info_list').find('li').find('p', class_='cast_txt').text.split(',')
    max_min_temp = soup.select('ul.info_list > li > span.merge')[0].text

    fine_dust = soup.select('#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.sub_info > div > dl > dd:nth-child(2)')[0].text
    ultra_fine_dust = soup.select('#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.sub_info > div > dl > dd:nth-child(4)')[0].text
    feeling_temper = soup.select('#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.main_info > div > ul > li:nth-child(2) > span.sensible')[0].text
    rainfall_per_hour = soup.select('#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.main_info > div > ul > li:nth-child(3) > span')[0].text
    number_feeling = soup.select('#main_pack > div.sc.cs_weather._weather > div:nth-child(2) > div.weather_box > div.weather_area._mainArea > div.today_area._mainTabContent > div.main_info > div > ul > li:nth-child(2) > span.sensible > em > span')[0].text

    if 'sys_location1' in body['action']['params']:
        sub_loc = body['action']['params']['sys_location1']
        if len(loc) <= 0:
            answer = '네트워크 접속에 문제가 발생하였습니다.'
        else:
            if float(number_feeling) < 20:
                answer = f'{loc} {sub_loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 쌀쌀할 수 있으니, \n 얇은외투를 챙기세요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            elif float(number_feeling) < 15:
                answer = f'{loc} {sub_loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 제법 쌀쌀하네요. \n외투를 챙기세요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            elif float(number_feeling) < 10:
                answer = f'{loc} {sub_loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 추워젔어요... \n옷을 두껍게 입는게 좋겠어요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            else:
                answer = f'{loc} {sub_loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 덥네요... \n얇게 입고 나가시는게 좋겠어요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
    else:
        if len(loc) <= 0:
            answer = '네트워크 접속에 문제가 발생하였습니다.'
        else:
            if float(number_feeling) < 20:
                answer = f'{loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 쌀쌀할 수 있으니, \n얇은 외투를 챙기세요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            elif float(number_feeling) < 15:
                answer = f'{loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 제법 쌀쌀하네요. \n외투를 챙기세요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            elif float(number_feeling) < 10:
                answer = f'{loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 추워젔어요... \n옷을 두껍게 입는게 좋겠어요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
            else:
                answer = f'{loc}의 날씨는 {weather1}, \n{weather2} \n\n현재 기온은 {temperature}도, \n최저 최고 기온은 {max_min_temp}, \n{feeling_temper} 입니다.\n\n날씨가 덥네요... \n얇게 입고 나가시는게 좋겠어요!\n\n미세먼지 수치는 {fine_dust} \n초미세먼지 수치는 {ultra_fine_dust}, \n\n오전 {am_rain_pct}, \n오후 {pm_rain_pct}, \n{rainfall_per_hour}입니다.'
    result = {
        "version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer}}]}
    }

    return jsonify(result)


@app.route('/song', methods=['GET', 'POST'])
def song():
    error_msg = '죄송하지만 다시 추천해 주세요 ㅠ'
    body = request.get_json()
    song = body['action']['params']['sys_text']

    if not '-' in song:
        answer = error_msg
        result = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer}}]}}

    else:
        artist, title = song.split('-')

        query = f'{artist} {title}'

        rec_list = ['추천해 주셔서 감사합니다!', '꼭 한번 들어볼게요!', '앞으로 더 많은 노래들을 추천해 주세요!']
        ans_list = ['이 노래 어떠신가요?', '이 노래 한번 들어보세요!', '누군가 한테 추천 받은 노래인데 노래 좋아요!', '이 노래 띵곡이에요!']
        url, thumbnail = search_youtube(query)

        fpath = '../DB/song_db.csv'
        db = pd.read_csv(fpath, encoding='utf-8', engine='python', names=['no', 'artist', 'answer', 'url', 'thumnail'])

        c_time = t.ctime()
        random.seed(c_time)

        db.loc[len(db) + 1] = [len(db) + 1, artist, random.sample(ans_list, 1)[0], url, thumbnail]
        db.drop_duplicates(['url'])
        db.to_csv(fpath, encoding='utf-8', header=False)

        answer = random.sample(rec_list, 1)
        result = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer[0]}}]}}

    return jsonify(result)


def search_youtube(query, number_of=1):
    key = google_key
    engine = 'youtube'
    version = 'v3'

    youtube = build(engine, version, developerKey=key)
    request = youtube.search().list(q=query, part='snippet', type='video', maxResults=number_of)
    response = request.execute()

    if number_of == 1:
        item = response['items']
        videoId = item[0]['id']['videoId']
        thumbnail = item[0]['snippet']['thumbnails']['medium']['url']
        url = 'https://youtube.com/watch?v=' + videoId
        title = item[0]['snippet']['title']

    else:
        item = response['items']
        url = ['https://youtube.com/watch?v=' + item[rep]['id']['videoId'] for rep in range(number_of)]
        thumbnail = [item[rep]['snippet']['thumbnails']['medium']['url'] for rep in range(number_of)]
        title = [item[rep]['snippet']['title'] for rep in range(number_of)]

    return url, thumbnail, title


@app.route('/song_reco', methods=['GET', 'POST'])
def song_reco():
    fpath = '../DB/song_db.csv'
    db = pd.read_csv(fpath, encoding='utf-8')

    c_time = t.ctime()
    random.seed(c_time)

    rand = random.randint(1, len(db))
    url = db['url'][rand]
    thumbnail = db['thumnail'][rand]
    answer = db['answer'][rand]

    result = {'version': '2.0', 'template':
        {'outputs': [
            {'basicCard': {
                'description': answer,
                'thumbnail': {'imageUrl': thumbnail},
                'buttons': [
                    {'label': '들으러가기',
                     'action': 'webLink',
                     'webLinkUrl': url}
                ]}
            }
        ]
        }}
    return jsonify(result)


@app.route('/dining', methods=['GET', 'POST'])
def dining():
    body = request.get_json()
    loc = body['action']['params']['sys_location']

    if 'dining' in body['action']['params']:
        dining = body['action']['params']['dining']

        if 'sys_location1' in body['action']['params']:
            sub_loc = body['action']['params']['sys_location1']
            query = loc + sub_loc + dining

        else:
            query = loc + dining
    else:
        if 'sys_location1' in body['action']['params']:
            sub_loc = body['action']['params']['sys_location1']
            query = loc + sub_loc
        else:
            query = loc
    error_msg = '찾으시는 결과가 부족하거나 없습니다.. ㅠ'
    try:

        name, cate, comments, addrs = search_dining(query)
        detail_addrs = [addrs[addr][1][:-7] for addr in range(len(addrs))]

        result = {'version': '2.0',
                  'template': {
                      'outputs': [{
                          'carousel': {
                              'type': 'basicCard',
                              'items': [
                                  {'title': f'이름 : {name[0]} \n분류 : {cate[0]}',
                                   'description': f'태그 : {comments[0]} \n\n 주소 \n {detail_addrs[0]}'},
                                  {'title': f'이름 : {name[1]} \n분류 : {cate[1]}',
                                   'description': f'태그 : {comments[1]} \n\n 주소 \n {detail_addrs[1]}'},
                                  {'title': f'이름 : {name[2]} \n분류 : {cate[2]}',
                                   'description': f'태그 : {comments[2]} \n\n 주소 \n {detail_addrs[2]}'},
                                  {'title': f'이름 : {name[3]} \n분류 : {cate[3]}',
                                   'description': f'태그 : {comments[3]} \n\n 주소 \n {detail_addrs[3]}'},
                                  {'title': f'이름 : {name[4]} \n분류 : {cate[4]}',
                                   'description': f'태그 : {comments[4]} \n\n 주소 \n {detail_addrs[4]}'},
                                  {'title': f'이름 : {name[5]} \n분류 : {cate[5]}',
                                   'description': f'태그 : {comments[5]} \n\n 주소 \n {detail_addrs[5]}'},
                                  {'title': f'이름 : {name[6]} \n분류 : {cate[6]}',
                                   'description': f'태그 : {comments[6]} \n\n 주소 \n {detail_addrs[6]}'},
                                  {'title': f'이름 : {name[7]} \n분류 : {cate[7]}',
                                   'description': f'태그 : {comments[7]} \n\n 주소 \n {detail_addrs[7]}'},
                                  {'title': f'이름 : {name[8]} \n분류 : {cate[8]}',
                                   'description': f'태그 : {comments[8]} \n\n 주소 \n {detail_addrs[8]}'},
                                  {'title': f'이름 : {name[9]} \n분류 : {cate[9]}',
                                   'description': f'태그 : {comments[9]} \n\n 주소 \n {detail_addrs[9]}'},

                              ]
                          }
                      }
                      ]
                  }}
    except:
        result = {
            "version": "2.0", "template": {"outputs": [{"simpleText": {"text": error_msg}}]}
        }

    return jsonify(result)


def search_dining(query):
    enc_query = str(urllib.parse.quote(query))
    url = 'http://www.diningcode.com/list.php?query=' + enc_query
    html = req.get(url).text

    soup = bs(html, 'lxml')

    name = [soup.select('#div_list > li > a > span.btxt')[rep].text.split('.')[1] for rep in range(10)]
    cate = [soup.select('#div_list > li > a > span.stxt')[rep].text for rep in range(10)]
    comments = [soup.select(f'#div_list > li:nth-child({rep}) > a > span:nth-child(4)')[0].text for rep in range(4, 14)]
    addrs = [str(soup.select(f'#div_list > li:nth-child({rep}) > a > span:nth-child(5)')[0]).split('</i>') for rep in
             range(4, 14)]

    return name, cate, comments, addrs


@app.route('/gag', methods=['GET', 'POST'])
def gag():
    db_path = '../DB/gag_db.csv'
    db = pd.read_csv(db_path, encoding='utf-8', names=['question', 'answer'])

    random.seed(t.ctime())
    rand = random.randint(1, len(db))
    q = db['question'][rand]
    a = db['answer'][rand]

    result = {
        "version": "2.0", "template": {"outputs": [{"simpleText": {"text": f'{q}? \n{a} ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'}}]}
    }

    return jsonify(result)


@app.route('/translate', methods=['GET', 'POST'])
def translate():
    body = request.get_json()
    block = body['userRequest']['block']['name']
    sentence = body['action']['params']['sys_text']

    key = 'Xbbwo5vSD6gCq_5Fa2P8'
    pw = 'hHisHjqN_M'

    if block == 'ko2zh-CN':
        encText = urllib.parse.quote(sentence)
        data = f'source={block[:2]}&target={block[-5:]}&text={encText}'
        url = "https://openapi.naver.com/v1/papago/n2mt"

    elif block == 'zh-CN2ko':
        encText = urllib.parse.quote(sentence)
        data = f'source={block[:5]}&target={block[-2:]}&text={encText}'
        url = "https://openapi.naver.com/v1/papago/n2mt"

    else:
        encText = urllib.parse.quote(sentence)
        data = f'source={block[:2]}&target={block[-2:]}&text={encText}'
        url = "https://openapi.naver.com/v1/papago/n2mt"
    error_msg = '지원하지 않는 언어입니다.'

    try:
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", key)
        req.add_header("X-Naver-Client-Secret", pw)
        response = urllib.request.urlopen(req, data=data.encode("utf-8"))

        response_body = response.read()
        papago = json.loads(response_body.decode('utf-8'))
        after = papago['message']['result']['translatedText']
        answer = f'번역 전 : {sentence} \n번역 후 : {after}'
        result = {
            "version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer}}]}
        }

    except:
        result = {
            "version": "2.0", "template": {"outputs": [{"simpleText": {"text": error_msg}}]}
        }

    return jsonify(result)


@app.route('/fortune', methods=['GET', 'POST'])
def fortune():
    body = request.get_json()
    query = body['action']['params']['sys_text']
    fortune = czs_fortune(query)
    result = {'version': '2.0',
              'template': {
                  'outputs': [{
                      'simpleText': {
                          'text': fortune
                      }
                  }]
              }}
    return jsonify(result)


@app.route('/horoscope_fortune', methods=['GET', 'POST'])
def horoscope_fortune():
    body = request.get_json()
    query = body['action']['params']['sys_text']
    birthday, fortune = horoscope(query)
    result = {'version': '2.0',
              'template': {
                  'outputs': [{
                      'simpleText': {
                          'text': f'생일 : {birthday} \n{fortune}'
                      }
                  }]
              }}
    return jsonify(result)


def czs_fortune(query):
    if (query[:-2] != '운세') or (query[:-3] != ' 운세'):
        query += '운세'

    enc_query = urllib.parse.quote(f'{query} 운세')
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=' + enc_query

    request = req.get(url)
    soup = bs(request.text, 'lxml')

    try:
        if query[:2] == '오늘':
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(3)')[0].text

        elif query[:2] == '내일':
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(4)')[0].text

        elif (query[:2] == '이주') or (query[:3] == '이번주'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(5)')[0].text

        elif (query[:2] == '이달') or (query[:3] == '이번달'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(6)')[0].text

        elif (query[:2] == '쥐띠') or (query[:2] == '소띠') or (query[:2] == '용띠') or (query[:2] == '뱀띠'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(3)')[0].text

        elif (query[:2] == '말띠') or (query[:2] == '양띠') or (query[:2] == '닭띠') or (query[:2] == '개띠'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(3)')[0].text

        elif (query[:3] == '토끼띠') or (query[:2] == '돼지띠'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(3)')[0].text

        elif (query[:4] == '호랑이띠') or (query[:4] == '원숭이띠'):
            fortune = soup.select('#yearFortune > div > div.detail > p:nth-child(3)')[0].text

        pattern = re.compile('\. ')
        fortune = re.sub(pattern, '. \n', fortune)

    except:
        fortune = '저는 오늘, 내일, 이주, 이번주, 이번달 이달만 알아들어요 ㅠ'

    return fortune


def horoscope(query):
    if (query[:-2] != '운세') or (query[:-3] != ' 운세'):
        query += '운세'

    enc_query = urllib.parse.quote(f'{query} 운세')
    url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=' + enc_query

    request = req.get(url)
    soup = bs(request.text, 'lxml')

    try:
        if query[:2] == '오늘':
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(3)')[0].text

        elif query[:2] == '내일':
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(4)')[0].text

        elif (query[:2] == '이주') or (query[:3] == '이번주'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(5)')[0].text

        elif (query[:2] == '이달') or (query[:3] == '이번달'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(6)')[0].text

        elif (query[:4] == '물병자리') or (query[:4] == '황소자리') or (query[:4] == '사자자리') or (query[:4] == '처녀자리'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(3)')[0].text

        elif (query[:4] == '천칭자리') or (query[:4] == '전갈자리') or (query[:4] == '사수자리') or (query[:4] == '염소자리'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(3)')[0].text

        elif (query[:3] == '양자리') or (query[:3] == '게자리'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(3)')[0].text

        elif (query[:5] == '물고기자리') or (query[:5] == '쌍둥이자리'):
            fortune = soup.select('#yearFortune > div > div.detail.detail2 > p:nth-child(3)')[0].text

        pattern = re.compile('\. ')
        fortune = re.sub(pattern, '. \n', fortune)

    except:
        fortune = '저는 오늘, 내일, 이주, 이번주, 이번달 이달만 알아들어요 ㅠ'

    birthday = soup.select('#yearFortune > div > div.thumb > span')[0].text
    return birthday, fortune


@app.route('/fortune_cookie', methods=['GET', 'POST'])
def fortune_cookie():
    cookie_path = '../DB/fortune_cookie.csv'
    data = pd.read_csv(cookie_path, engine='python', encoding='utf-8', names=['no', 'fortune', 'speaker'])

    fortune = [data['fortune'][rep].replace('.', '. \n') for rep in range(1, len(data))]
    speaker = [data['speaker'].fillna('화자 미상', inplace=True)]
    speaker = [data['speaker'][rep] for rep in range(1, len(data))]

    c_time = t.ctime()
    random.seed(c_time)

    rand = random.randint(1, len(data) - 1)
    fortune = fortune[rand]

    speaker = speaker[rand]
    res = fortune + '- ' + speaker

    result = {'version': '2.0',
              'template': {
                  'outputs': [{
                      'simpleText': {
                          'text': res
                      }
                  }]
              }}

    return jsonify(result)


@app.route('/experi', methods=['GET', 'POST'])
def experi():
    body = request.get_json()
    body = body['userRequest']

    result = {
        "version": "2.0", "template": {"outputs": [{"simpleText": {"text": body['utterance']}}]}
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(sys.argv[1]), threaded=True, debug=True)





from glob import glob
import re
import os

path = '../AI_reboot/Keras/Datas/RNN/kakao_talk_dialog'
text_path = glob(os.path.join(path, '*.txt'))

from pprint import pprint
text_lis = [open(text, 'r', encoding='utf-8').read() for text in text_path]
texts = ''
for lines in text_lis:
    texts += str(lines)

texts = texts.replace('[','')
texts = texts.replace(']','')
texts = texts.replace("\\ufeff", '')
texts = texts.replace("\ufeff", '')

date_am = re.compile(r'\d{4}년 \d{1,2}월 \d{1,2}일 오전 \d{1,2}:\d{2},')
date_pm = re.compile(r'\d{4}년 \d{1,2}월 \d{1,2}일 오후 \d{1,2}:\d{2},')
date = re.compile(r'\d{4}년 \d{1,2}월 \d{1,2}일 오후 \d{1,2}:\d{2}\n')
photo = re.compile(r' [가-힣]+ : 사진\n')
emoji = re.compile(r' [가-힣]+ : 이모티콘\n')
emoji2 = re.compile(r' [가-힣]+ : 이모티콘\\n')
toss = re.compile(r' [가-힣]+ : 토스 행운퀴즈: 퀴즈를 풀면 행운상금이 쏟아집니다!\n')
toss2 = re.compile(r'https://toss.im/_m/[a-zA-Z0-9]+\n')

texts = re.sub(date_am, '', texts)
texts = re.sub(date_pm, '', texts)
texts = re.sub(photo, '', texts)
texts = re.sub(emoji, '', texts)
texts = re.sub(emoji2, '', texts)
texts = re.sub(date, '', texts)
texts = re.sub(toss, '', texts)
texts = re.sub(toss2, '', texts)


pprint(texts)
print(type(texts))
with open('kakao_dialog.txt', 'w+', encoding='utf-8') as f:
    f.writelines(texts)

from text_prepro import *
import pandas as pd
import numpy as np


PAD = "<PADDING>"
STA = "<START>"
END = "<END>"
OOV = "<OOV>"

PAD_IDX = 0
STA_IDX = 1
END_IDX = 2
OOV_IDX = 3

E_input = 0
D_input = 1
D_target = 2

max_len, embedding_dim = 30, 100
lstm_h_dim = 128


f_path = './datas/text/ChatbotData.csv'
sents_data = pd.read_csv(f_path, engine = 'python')
q, a = list(sents_data['Q']), list(sents_data['A'])

q, a = q[:10640], a[:10640]
q = pos_tag(q)
a = pos_tag(a)

sents = []
sents.extend(q)
sents.extend(a)

words = []
for sent in sents:
    for word in sent.split():
        words.append(word)

words = [word for word in words if len(word) >0]
words = list(set(words))
words[:0] = [PAD, STA, END, OOV]

word_2_idx = {word : idx for idx, word in enumerate(words)}
idx_2_word = {idx : word for idx, word in enumerate(words)}
dict(list(word_2_idx.items())[:])

x_encoder = text_2_idx(q, word_2_idx, E_input)
x_decoder = text_2_idx(a, word_2_idx, D_input)
y_decoder = text_2_idx(a, word_2_idx, D_input)

from keras import models
w_path = './s2s.h5'
m = models.load_model(w_path)
layers = dict([(layer.name, layer.output) for layer in m.layers])

def make_predict_input(sentence):
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = text_2_idx(sentences, word_2_idx, E_input)

    return input_seq

def generate_text(input_seq):
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))

    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX

    # 인덱스 초기화
    indexs = []

    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
            [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)

        # 종료 검사
        if index == END_IDX or len(indexs) >= max_len:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index

        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = idx_2_text(indexs, idx_2_word)

    return sentence

encoder_inputs = layers['input_23']
_, state_h, state_c = layers['lstm15']
encoder_state = [state_h, state_c]
encoder_model = models.Model(encoder_inputs, encoder_state)

decoder_state_h =










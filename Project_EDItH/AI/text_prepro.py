# -*- coding : utf-8 -*-

import numpy as np
from konlpy.tag import Okt
import re


# NOTE : 태그 단어
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

# NOTE : 형태소 분석 함수
def pos_tag(sents):

    tagger = Okt()
    sent_pos = []

    filter = re.compile("[.,!?\"':;~()]")
    for sent in sents:

        sent = re.sub(filter, '', sent)
        sent = ''.join(tagger.morphs(sent))
        sent_pos.append(sent)

    return sent_pos

# NOTE : 문장을 인덱스로 변환시켜줌.
def text_2_idx(sents, voca, type):

    sents_idx = []
    for sent in sents:
        sent_idx = []

        if type == D_input:
            sent_idx.extend([voca[STA]])

        for word in sent.split():
            if voca.get(word) is not None:
                # NOTE : 사전에 있는 단어면 해당 인덱스 추가
                sent_idx.extend([voca[word]])
            else:
                # NOTE : 사전에 없는 단어면 OOV인덱스 추가
                sent_idx.extend([voca[OOV]])

        # NOTE : 최대 길이 검사
        if type == D_target:
            if len(sent_idx) >= max_len:
                sent_idx = sent_idx[:max_len-1] + [voca[END]]

            else:
                sent_idx += [voca[END]]

        else:
            if len(sent_idx) > max_len:
                sent_idx = sent_idx[:max_len]

        sent_idx += (max_len - len(sent_idx))*[voca[PAD]]
        sents_idx.append(sent_idx)
    return np.asarray(sents_idx)

# NOTE : 인덱스를 문장으로 변환시켜줌.
def idx_2_text(idxs, voca):

    sent = ''

    for idx in idxs:
        if idx == END_IDX:
            break

        if voca.get(idx) is not None:
            sent += voca[idx]

        else:
            sent.extend([voca[OOV_IDX]])

        sent += ' '
    return sent


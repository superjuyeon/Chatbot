import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import okt

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)
MAX_SEQUENCE = 255

# 데이터 불러오기
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer

# 단어 리스트
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER,'',sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# 형태소 분리
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = ' '.join(morph_analyzer.morphs(seq.replace(' ','')))
        result_data.append(morphlized_seq)

    return result_data

#단어 사전 만들기
def load_vocabulary(path, vocab_path):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        data_df = pd.read_csv(path, encoding='utf-8')
        question, answer = list(data_df['Q']), list(data_df['A'])
        data=[]
        data.extend(question)
        data.extend(answer)

        words = data_tokenizer(data)
        words = list(set(words))
        words[:0] = MARKER

        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word+'\n')

    with open(vocab_path,'r',encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocabulary_list)

    return word2idx, idx2word, len(word2idx)


# 각각 단어에 대한 인덱스와 인덱스에 대한 단어를 가진 딕셔너리 데이터를 뽑아주는 함수
def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스고 값이 단어인 딕셔너리
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}

    return word2idx, idx2word


word2idx, idx2word, vocab_size = load_vocabulary(PATH, VOCAB_PATH)

# 인코더에 적용될 입력값을 만드는 전처리 함수
def enc_preprocessing(value, dictionary):
    sequences_input_index=[]
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER,'',sequence)
        sequence_index = []

    for word in sequence.split():
        if dictionary.get(word) is not None:
            sequence_index.extend([dictionary[word]])
        else: 
            sequence_index.extend([dictionary[UNK]])

    if len(sequence_index) > MAX_SEQUENCE:
        sequence_index = sequence_index[:MAX_SEQUENCE]

    sequences_length.append(len(sequence_index))
    sequence_index += (MAX_SEQUENCE - len(sequence_index)) * dictionary[PAD]

    sequences_input_index.append(sequence_index)

return np.asarray(sequence_input_index), sequences_length

# 디코더의 입력값을 만드는 함수
def dec_output_preprocessing(value, dictionary):
    sequences_output_index = []
    sequences_length = []

    for sequence in value:
        sequences = re.sub(CHANGE_FILTER,'',sequence)
        sequence_index = []
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE- len(sequence_index)) * [dictionary[PAD]]

        sequences_output_index.append(sequence_index)

    return np.asarray(sequences_output_index), sequences_length

# 디코더의 타깃값을 만드는 함수 
def dec_target_processing(value, dictionary):
    sequences_target_index = []
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER,'',sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        
        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE -1] + [dictionary[END]]

        else: 
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)
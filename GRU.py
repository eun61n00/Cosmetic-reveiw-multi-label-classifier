import pandas as pd
import numpy as np
import konlpy
import os, time, pickle

df = pd.read_csv('prd_review_df.csv')
df.drop_duplicates(subset='review_content', inplace=True)

tokenizer = konlpy.tag.Okt().morphs

def build_vocab(docs, min_len=2, stopwords=None, tokenizer=tokenizer):
    words = set()
    i = 0
    for doc in docs:
        words |= {token for token in tokenizer(doc) if len(token) >= min_len}
        i += 1
        if i % 1000 == 0:
            print(f'document {i}/{len(docs)} complete!')
    if stopwords is not None:
        words -= set(stopwords)
    vocab = {words: idx for idx, word in enumerate(words)}
    return vocab

with open('stopwords_ko.pkl', 'rb') as f:
    stopwords_ko = pickle.load(f)

vocab = build_vocab(df['review_content'], stopwords=stopwords_ko, tokenizer=tokenizer)

words = set()
i = 0
for sentence in df['review_content']:
    words |= {token for token in tokenizer(sentence) if len(token) >= 2}
    i += 1
    if i % 1000 == 0:
            print(f'document {i}/{len(df.review_content)} complete!')
words -= set(stopwords_ko)
words = list(words)
vocab = {}
for idx, word in enumerate(words):
    vocab[word] = idx

content_all = ' '.join(df.review_content)
rare_word = []
for i, rare in enumerate(list(vocab.keys())):
    if content_all.count(rare) <= 3:
        rare_word.append(rare)
    if i % 1000 == 0:
        print(f'{i}/{len(list(vocab.keys()))} complete!')

vocab_v2 = {k:v for k,v in vocab.items() if k not in rare_word}

print('리뷰의 최대 길이: ', max(len(l) for l in df['review_content']))
print('리뷰의 평균 길이: ', sum([len(l) for l in df['review_content']]) / len(df.review_content))

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt += 1
    print('전체 리뷰 중 길이가 %s 이하인 리뷰의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 35
below_threshold_len(70, df.review_content)

from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(df.review_content, maxlen=70)
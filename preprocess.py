import pandas as pd
import numpy as np
import pickle
import re
import os

DATA_PATH = os.getcwd() + '/data'

def remove_linebreak(string):
    return string.replace('\r',' ').replace('\n',' ')

def split_table_string(string):
    trimmedTableString=string[string.rfind("PNR :"):]
    string=string[:string.rfind("PNR :")]
    return (string, trimmedTableString)

def remove_multispace(x):
    x = str(x).strip()
    x = re.sub(' +', ' ',x)
    return x

def flatten(lst):
    return [item for sublist in lst for item in sublist]

class preprocess:
    def __init__(self):
        self.prd_review_dict = pickle.load(open(f'{DATA_PATH}/prd_dict.pkl', 'rb'))
        self.prd_review_df = None
        self.review = []
        self.target = []
        self.target_names = []

    def make_dataframe(self):

        ratings_lst = []
        highlight_lst = []
        review_contents_lst = []
        topic_lst = []

        for i in range(len(list(self.prd_review_dict.keys()))):
            dict_keys_lst = list(self.prd_review_dict.keys())
            for topic in list(self.prd_review_dict[dict_keys_lst[i]].keys()):
                ratings_lst.append(self.prd_review_dict[dict_keys_lst[i]][topic][0])
                topic_lst.append([topic for _ in range(len(self.prd_review_dict[dict_keys_lst[i]][topic][0]))])
                highlight_lst.append(self.prd_review_dict[dict_keys_lst[i]][topic][1])
                review_contents_lst.append(self.prd_review_dict[dict_keys_lst[i]][topic][2])

        ratings_lst = flatten(ratings_lst);
        highlight_lst = flatten(highlight_lst)
        review_contents_lst = flatten(review_contents_lst);
        topic_lst = flatten(topic_lst);

        d = {'ratings': ratings_lst,
             'highlight': highlight_lst,
             'topic': topic_lst,
             'review_content': review_contents_lst}

        prd_review_df = pd.DataFrame(data=d)

        # group by 'review content'
        group_highlight = prd_review_df['highlight'].groupby([prd_review_df.review_content]).apply(list).reset_index()
        group_topic = prd_review_df['topic'].groupby([prd_review_df.review_content]).apply(list).reset_index()
        ratings = prd_review_df['ratings'].groupby([prd_review_df.review_content]).mean().reset_index()

        # merge
        group = pd.merge(group_highlight, group_topic, how='inner', on='review_content')
        prd_review_df = pd.merge(group, ratings, how='inner', on='review_content')

        # remove duplicated elements
        prd_review_df.highlight = prd_review_df.highlight.apply(lambda x: list(set(x)))
        prd_review_df.topic = prd_review_df.topic.apply(lambda x: list(set(x)))
        sentiment = [1 if rating >= 4 else 0 for rating in prd_review_df.ratings]
        prd_review_df['sentiment'] = sentiment

        df_topic_name = [topic.replace('향기/냄새', '향기').replace('사용성', '사용감')
                         for topic in flatten(list(prd_review_df.topic))]

        # represent topic as matrix
        topic_dict = {}
        i = 0
        for topic in set(df_topic_name):
            if df_topic_name.count(topic) / len(df_topic_name) > 0.005:
                topic_dict[topic] = i
                i += 1

        multi_label = np.zeros((len(prd_review_df), len(topic_dict)+1))
        self.target = []
        self.target_names = list(topic_dict.keys())
        self.target_names.append('sentiment')
        self.review = list(prd_review_df['review_content'])
        for i in range(len(prd_review_df)):
            for j in range(len(prd_review_df.iloc[[i]].topic.values[0])):
                if prd_review_df.iloc[[i]].topic.values[0][j] not in topic_dict.keys():
                    continue
                multi_label[i][topic_dict[prd_review_df.iloc[[i]].topic.values[0][j]]] = 1
            multi_label[i][len(topic_dict)] = prd_review_df.iloc[[i]].sentiment.values
            self.target.append(multi_label[i])
        self.prd_review_df = prd_review_df
        return True
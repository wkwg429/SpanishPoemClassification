# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为doc2vec特征，并将结果保存至本地
@author: Jian
"""
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle
import mlconfig as mlcfg

t_start = time.time()

"""=====================================================================================================================
0 辅助函数 
"""


def sentence2list(sentence):
    s_list = sentence.strip().split()
    return s_list


"""=====================================================================================================================
1 读取原始数据，并进行简单处理
"""
df_train = pd.read_csv(mlcfg.data_train_path)
df_test = pd.read_csv(mlcfg.data_test_path)
df_train['genre'].replace({'古典抒情': 0, '浪漫主义': 1, '现代主义': 2}, inplace=True)
# df_test['genre'].replace({'Parnasse': 0, 'romantisme': 1, 'Symbolisme': 2}, inplace=True)
df_test['genre'].replace({'古典抒情': 0, '浪漫主义': 1, '现代主义': 2}, inplace=True)
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = df_train['genre'].values
y_test = df_test['genre'].values

df_all['word_list'] = df_all['content'].apply(sentence2list)
texts = df_all['word_list'].tolist()

"""=====================================================================================================================
2 doc2vec
"""
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, vector_size=300, window=5, min_count=3, workers=4, epochs=25)
docvecs = model.dv

x_train = []
# for i in range(0, 1152):
for i in range(0, 198):
    x_train.append(docvecs[i])
x_train = np.array(x_train)

x_test = []
# for j in range(1152, 1646):
for j in range(198, 283):
    x_test.append(docvecs[j])
x_test = np.array(x_test)

"""=====================================================================================================================
3 将doc2vec特征保存至本地
"""
data = (x_train, y_train, x_test, y_test)
f_data = open('./spanish_feature/data_doc2vec.pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("已将原始数据数字化为doc2vec特征，共耗时：{}min".format((t_end-t_start)/60))

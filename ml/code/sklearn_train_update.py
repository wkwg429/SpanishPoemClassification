# -*- coding: utf-8 -*-

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
from sklearn_config import features_path, clf_name, clfs, status_vali
from sklearn.metrics import classification_report
import os
from sklearn import metrics

t_start = time.time()

features_name = ['data_tfidf.pkl', 'data_doc2vec.pkl']
# 'data_tf.pkl',
clf_name = ['lr', 'svm', 'bagging', 'rf', 'adaboost', 'gbdt', 'xgb', 'lgb']
# clf_name = ['xgb']
"""=====================================================================================================================
1 读取数据
"""
allresult = []

for feature in features_name:
    features_path = os.path.join(r'../features/spanish_feature', feature)
    print(features_path)
    results = []
    print('目前使用的特征是：' + feature + '-------------------------')
    data_fp = open(features_path, 'rb')
    x_train, y_train, x_test, y_test = pickle.load(data_fp)
    data_fp.close()
    # print('---------x_train---------')
    # print(x_train)
    # print('---------y_train---------')
    # print(y_train)
    feature = feature.replace('data_','').replace('.pkl','')
    print(feature)

    """划分训练集和验证集，验证集比例为test_size"""
    if status_vali:
        x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    """=====================================================================================================================
    2 训练分类器
    """
    for classification_name in clf_name:
        print('目前使用的分类算法是：' + classification_name +'----------------')
        clf = clfs[classification_name]
        clf.fit(x_train, y_train)

        """=====================================================================================================================
        3 在验证集上评估模型
        """
        if status_vali:
            pre_vali = clf.predict(x_vali)
            score_vali = f1_score(y_true=y_vali, y_pred=pre_vali, average='macro')
            print("验证集分数：{}".format(score_vali))

        """=====================================================================================================================
        4 对测试集进行预测评估
        """
        y_pred = clf.predict(x_test)

        pre_file_name = feature + '_' + classification_name + '.csv'
        output_file = os.path.join("../results/spanish/12_5", pre_file_name)
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['真实标签', '预测标签']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入文件头部
            writer.writeheader()

            # 遍历每个样本，写入真实标签和预测标签
            for true_label, predicted_label in zip(y_test, y_pred):
                writer.writerow({'真实标签': true_label, '预测标签': predicted_label})


        # 以下是分类算法评估代码
#         accuracy = metrics.accuracy_score(y_test, y_pred)
#         precision = metrics.precision_score(y_test, y_pred, average='macro')
#         recall = metrics.recall_score(y_test, y_pred, average='macro')
#         f1_score = metrics.f1_score(y_test, y_pred, average='macro')
#         report = classification_report(y_test, y_pred)
#         print('测试集评估结果：------------------------')
#         print(report)
#         t_end = time.time()
#         print("特征:"+ feature + "分类算法为:" + classification_name +"训练结束，耗时:{}min".format((t_end - t_start) / 60))
#         result = {
#             'Feature': feature,
#             'Classification': classification_name,
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'F1 Score': f1_score
#         }
#         results.append(result)
#         allresult.append(result)
#
#     # 将结果保存到CSV文件
#     filename = feature + '.csv'
#     result_file_path = os.path.join('../results/spanish/12_5', filename)
#     with open(result_file_path, 'w', newline='') as file:
#         writer = csv.DictWriter(file,
#                                 fieldnames=['Feature', 'Classification', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
#         writer.writeheader()
#         writer.writerows(results)
#
#     print('评估结果已保存至{}'.format(result_file_path))
# with open('../results/spanish/12_5/all.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file,
#                             fieldnames=['Feature', 'Classification', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
#     writer.writeheader()
#     writer.writerows(allresult)

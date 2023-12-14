# import os
# import csv
# import random
# import re
#
# import pandas as pd
#
#
# def process_folder(folder_path, genre):
#     """
#     读取一个指定路径下所有txt文件，并返回其内容
#     """
#     # 获取目标文件夹下所有的txt文件
#     txt_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.txt')]
#
#     # 遍历每个txt文件，读取内容后添加到列表中
#     contents = []
#     for file_path in txt_files:
#         with open(file_path, 'r', encoding='utf-8') as rf:
#             lines = rf.readlines()
#             lines = lines[3:]  # 删除前三行
#             lines = [line.strip() for line in lines if line.strip()]  # 去掉空行并保存
#             content = ' '.join(lines)  # 将每个txt文件中的内容合并为一行
#             # 去掉多余的空格和标点符号
#             content = re.sub(r'[^\w\s]', '', content)
#             content = re.sub(r'\s+', ' ', content)
#             contents.append(content)
#
#     # 返回各个txt文件的内容及对应的文学风格
#     return [(genre, content) for content in contents]
#
# def spilt_dataset():
#     df = pd.read_csv(r'C:\Users\King\Desktop\french_poems\output.csv', encoding='utf-8')
#
#     labels = set(df['genre'])
#     contents = df['content']
#
#     count = {}
#     cal = {}
#     for p in df['genre']:
#         cal[p] = 0
#         try:
#             count[p] += 1
#         except KeyError:
#             count[p] = 1
#     print(count)
#     train, test = [], []
#     for i, label in enumerate(df['genre']):
#         if cal[label] < count[label] * 0.7:
#             train.append({'label': label, 'content': contents[i]})
#         else:
#             test.append({'label': label, 'content': contents[i]})
#         cal[label] += 1
#     with open(r'C:\Users\King\Desktop\french_poems\train_set.csv', 'w', newline='', encoding='utf-8') as f:
#         xieru = csv.DictWriter(f, ['label', 'content'])
#         xieru.writeheader()
#         xieru.writerows(train)  # writerows方法是一下子写入多行内容
#     with open(r'C:\Users\King\Desktop\french_poems\test_set.csv', 'w', newline='', encoding='utf-8') as f:
#         xieru = csv.DictWriter(f, ['label', 'content'])
#         xieru.writeheader()
#         xieru.writerows(test)  # writerows方法是一下子写入多行内容
#
# def shuffle_csv(filename):
#     # 读取csv文件
#     with open(filename, 'r', newline='', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         data = list(reader)
#
#     # 打乱数据顺序
#     random.shuffle(data[1:])
#
#     # 将打乱后的数据写回csv文件
#     with open(filename, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerows(data)
#
# def main():
#     # 指定总文件夹路径和csv文件路径
#     base_folder = r'C:\Users\King\Desktop\french_poems'
#     csv_file = r'C:\Users\King\Desktop\french_poems\output.csv'
#
#     # 获取三个风格子文件夹路径
#     style_folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if
#                      os.path.isdir(os.path.join(base_folder, f))]
#
#     # 打开输出csv文件，指定文件路径和编码格式
#     with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#         # 创建csv写入对象
#         writer = csv.writer(f)
#
#         # 添加标题行
#         writer.writerow(['genre', 'content'])
#
#         # 遍历每个风格子文件夹，读取所有txt文件并写入csv文件
#         for folder in style_folders:
#             genre = os.path.basename(folder)
#             contents = process_folder(folder, genre)
#             for content in contents:
#                 writer.writerow(content)
#
#     spilt_dataset()
#     shuffle_csv(r'C:\Users\King\Desktop\french_poems\train_set.csv')
#     shuffle_csv(r'C:\Users\King\Desktop\french_poems\test_set.csv')
#
#     print('Done!')
#
#
# if __name__ == '__main__':
#     main()


import os
import csv
import re
import random

import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split


def read_data(folder_path):
    """
    读取一个指定路径下所有txt文件，并返回其内容
    """
    # 获取目标文件夹下所有的txt文件
    txt_files = [os.path.join(folder_path, fname)
                 for fname in os.listdir(folder_path)
                 if fname.endswith('.txt')]

    # 遍历每个txt文件，读取内容后添加到列表中
    contents = []
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            lines = [line.strip() for line in lines if line.strip()]  # 去掉空行并保存
            # lines = lines[3:]  # 删除前三行
            content = ' '.join(lines)  # 将每个txt文件中的内容合并为一行

            # 去掉多余的空格和标点符号
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'[^\w\s]', '', content)

            contents.append(content)

    # 得到该文化风格下所有的分词结果
    docs = []
    for content in contents:
        tokens = simple_preprocess(content, deacc=True)
        docs.append(tokens)

    # 返回各个txt文件的分词结果及对应的文学风格
    return [(os.path.basename(folder_path), tokens) for tokens in docs]


def write_data_to_csv(data, filename):
    """
    将数据写入csv文件
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['genre', 'content'])
        writer.writerows(data)


def main():
    # 指定总文件夹路径
    base_folder = '../first_process/spanishpoems'

    # 获取三个风格子文件夹路径
    style_folders = [os.path.join(base_folder, f)
                     for f in os.listdir(base_folder)
                     if os.path.isdir(os.path.join(base_folder, f))]

    # 遍历每个风格子文件夹，读取所有txt文件
    docs = []
    for folder in style_folders:
        contents = read_data(folder)
        docs += contents

    # 随机划分数据集
    data = [(tokens, genre)
            for genre, tokens in docs]
    random.shuffle(data)

    train_data, test_data = train_test_split(data, test_size=0.3)

    train_set = [(genre, ' '.join(tokens))
                 for tokens, genre in train_data]
    test_set = [(genre, ' '.join(tokens))
                for tokens, genre in test_data]

    # 将训练集和测试集写入到csv文件中
    write_data_to_csv(train_set, '../first_process/spanish_train_set.csv')
    write_data_to_csv(test_set, '../first_process/spanish_test_set.csv')

    print('Done!')


if __name__ == '__main__':
    main()
    # df = pd.read_csv('../first_process/train_set.csv')
    # print(df.head(10))
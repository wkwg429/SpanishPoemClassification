import pandas as pd
import matplotlib.pyplot as plt

def draw():
    # 读取CSV文件
    df = pd.read_csv('../ml/results/tfidf.csv')

    # 单独提取每一列数据
    classification = df['Classification']
    accuracy = df['Accuracy']
    precision = df['Precision']
    recall = df['Recall']
    f1_score = df['F1 Score']

    # 绘制折线图
    plt.plot(classification, accuracy, label='Accuracy')
    plt.plot(classification, precision, label='Precision')
    plt.plot(classification, recall, label='Recall')
    plt.plot(classification, f1_score, label='F1 Score')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Classification')
    plt.ylabel('Scores')

    plt.ylim(0.45, 0.8)

    # 显示图形
    plt.show()

if __name__ == "__main__":
    draw()
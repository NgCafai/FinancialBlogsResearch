import os
import pymysql
from settings import *
from collections import Counter
import pickle
from tensorflow import keras
import numpy
from typing import Dict

def get_synonyms(file_path):
    """
    返回同义词字典
    :return: dict[str, str]
    """
    dic = {}
    path = os.path.join(file_path, './synonyms.txt')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split(' ')
            target = words[0].strip()
            for i in range(1, len(words)):
                dic[words[i].strip()] = target
    return dic


def get_stop_words(file_path) -> set:
    """
    返回停用词表
    :return:
    """
    stop_words_dir = os.path.join(file_path, './stop_words.txt')
    with open(stop_words_dir, 'r', encoding='utf-8') as f:
        stop_words: set = {word.strip() for word in f if word.strip()}
    return stop_words


def build_vocab(vocab_dir='./vocabulary.txt'):
    """
    根据2009年初到2017年末的文本（分词后），选出频率最高的词，构建词典，并保存到vocab_dir中
    :param vocab_dir:
    :param vocab_size:
    :return:
    """
    all_data = []
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT)
    cursor = db.cursor()

    # 构建查询语句，并获取结果
    sql = "select * from processed_blogs " \
          "where created_date between '2008-01-01 00:00:00' and '2018-01-01 00:00:00'"
    try:
        cursor.execute(sql)
        processed_blogs = cursor.fetchall()
    except:
        print('Error when fetching blogs')
        return

    for processed_blog in processed_blogs:
        all_data.extend(processed_blog[2].split())

    # 选出vocab - 1个频率最高的词
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    most_frequent_words, _ = list(zip(*count_pairs))
    most_frequent_words = ['<pad>'] + list(most_frequent_words)
    # 写入到词汇表
    with open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(most_frequent_words) + '\n')


def read_vocab(file_path):
    """
    读取词汇表，并返回包含所有词汇的list，以及一个包含 词汇：编码 的字典
    :param file_path: 词典所在的文件夹路径
    :return:
    """
    path = os.path.join(file_path, './vocabulary.txt')
    with open(path, 'r', encoding='utf-8') as f:
        words = [x.strip() for x in f.readlines() if x.strip()]

    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def get_date_returns(file_path: str) -> dict:
    path = os.path.join(file_path, './date.pk')
    with open(path, 'rb') as f:
        date = pickle.load(f)

    path = os.path.join(file_path, './next_three_day_returns.pk')
    with open(path, 'rb') as f:
        next_three_day_returns = pickle.load(f)

    date_returns = {k: v for k, v in zip(date, next_three_day_returns)}
    return date_returns


def get_all_samples(date_returns: dict, word_to_id: dict):
    """
    返回训练集和测试集，用id表示
    :return:
    """
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT, charset='utf8')
    cursor = db.cursor()

    # 获取训练集所有数据
    train_blogs = []
    for blogger_name in blogger_names:
        sql = "select * from processed_blogs " \
              "where blogger_name = \'%s\' and created_date between \'%s\' and \'%s\' order by created_date desc;" \
              % (blogger_name, train_start, train_end)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            raise Exception("Error when fetching blogs")
        train_blogs.extend(results)  # 每一个元素是一个tuple，代表了数据库中的一行

    # 获取测试集所有数据
    test_blogs = []
    for blogger_name in blogger_names:
        sql = "select * from processed_blogs " \
              "where blogger_name = \'%s\' and created_date between \'%s\' and \'%s\' order by created_date desc;" \
              % (blogger_name, test_start, test_end)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            raise Exception("Error when fetching blogs")
        test_blogs.extend(results)  # 每一个元素是一个tuple，代表了数据库中的一行

    # 构建训练集
    x_train: list[list] = []
    y_train: list[int] = []
    for blog in train_blogs:
        words: list[str] = blog[2].split(' ')
        created_date: datetime.datetime = blog[3]
        words_id: list[int] = [word_to_id[word] for word in words if word in word_to_id.keys()]
        # 判断是否大于预设的seq_length，是的话就截取
        if len(words_id) > seq_length:
            mid = int(len(words_id) / 2)
            words_id = words_id[0: int(seq_length / 3)] \
                       + words_id[mid - int(seq_length / 6): mid + int(seq_length / 6)] \
                       + words_id[-int(seq_length / 3):]
        x_train.append(words_id)

        # 根据博文的发布时间，找到对应的标签
        date = created_date.date()
        time = created_date.time()
        # 若博文是在早上9：25前发布的，date推前一天
        if time.__lt__(datetime.time(9, 25, 0, 0)):
            date = date + datetime.timedelta(days=-1)
        # 若date为休市日，推前一天
        while date not in date_returns.keys():
            date = date + datetime.timedelta(days=-1)
        next_three_day_return = date_returns[date]
        if next_three_day_return < NEGATIVE_BOUNDARY:
            y_train.append(0)
        elif next_three_day_return > POSITIVE_BOUNDARY:
            y_train.append(2)
        else:
            y_train.append(1)

    # 构建测试集
    x_test: list[list] = []
    y_test: list[int] = []
    for blog in test_blogs:
        words: list[str] = blog[2].split(' ')
        created_date: datetime.datetime = blog[3]
        words_id: list[int] = [word_to_id[word] for word in words if word in word_to_id.keys()]
        # 判断是否大于预设的seq_length，是的话就截取
        if len(words_id) > seq_length:
            mid = int(len(words_id) / 2)
            words_id = words_id[0: int(seq_length / 3)] \
                       + words_id[mid - int(seq_length / 6): mid + int(seq_length / 6)] \
                       + words_id[-int(seq_length / 3):]
        x_test.append(words_id)

        # 根据博文的发布时间，找到对应的标签
        date = created_date.date()
        time = created_date.time()
        # 若博文是在早上9：25前发布的，date推前一天
        if time.__lt__(datetime.time(9, 25, 0, 0)):
            date = date + datetime.timedelta(days=-1)
        # 若date为休市日，推前一天
        while date not in date_returns.keys():
            date = date + datetime.timedelta(days=-1)
        next_three_day_return = date_returns[date]
        if next_three_day_return < NEGATIVE_BOUNDARY:
            y_test.append(0)
        elif next_three_day_return > POSITIVE_BOUNDARY:
            y_test.append(2)
        else:
            y_test.append(1)

    # 将x_train中的元素pad为固定长度max_length
    x_train: numpy.array = keras.preprocessing.sequence.pad_sequences(x_train, seq_length, padding='post')
    # 将y_train中的元素转换为one_hot表示
    y_train: numpy.array = keras.utils.to_categorical(y_train, num_classes=3)

    x_test = keras.preprocessing.sequence.pad_sequences(x_test, seq_length, padding='post')
    y_test = keras.utils.to_categorical(y_test, num_classes=3)

    return x_train, y_train, x_test, y_test


def batch_iter(x, y_):
    """
    生成器：每迭代一次，生成一个batch的数据
    :param x:
    :param y_:
    :return:
    """
    data_len = len(x)
    num_batch = int(float(data_len - 1) / float(batch_size)) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min(start_id + batch_size, data_len)
        yield x[start_id:end_id], y_[start_id:end_id]


# if __name__ == '__main__':
    # dic = get_synonyms('./')
    # stop_words = get_stop_words('./')
    # print(type(stop_words))
    # print(',' in stop_words)
    # print('市场' in dic.keys())
    # print(dic['上升'] == '上涨')
    # for k, v in dic.items():
    #     print(k, v)

    # build_vocab()

    # results = get_all_samples(['余岳桐'])
    # print(results[3])
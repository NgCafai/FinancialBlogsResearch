import pymysql
from settings import *
import jieba
import jieba.analyse
import os
import re
from Util.utils import *


def process_original_blogs(blogger_name):
    """
    对于某个博主的博文进行处理，包括分词、选择重点词汇、去除停用词、同义词处理，
    然后输出到MySQL数据库中
    :param blogger_name:
    :return:
    """
    # 加载自己整理的词典
    jieba.load_userdict("../../Util/MyDict.txt")
    # 连接数据库
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT)
    cursor = db.cursor()

    # 停用词表
    stop_words = get_stop_words('../../Util/')
    # 同义词字典
    synonyms_dic = get_synonyms('../../Util/')

    # 构建查询语句，并获取结果
    sql = 'select * from blogs where blogger_name = \'%s\' order by created_date desc;' % blogger_name
    try:
        cursor.execute(sql)
        blogs = cursor.fetchall()
    except:
        print('Error when fetching blogs')
        return

    # 测试
    # f = open('./test.txt', 'w', encoding='utf-8', errors='ignore')
    # sentence = blogs[633][3].replace("\n", "")
    # print(blogs[633][-1])
    # seg_list = jieba.cut(sentence)
    # for word in seg_list:
    #     f.write(word + "\n")
    # f.close()
    # key_words = jieba.analyse.textrank(sentence, topK=50, withWeight=True)
    # for word, weight in key_words:
    #     print(word, weight)

    # 处理文本并写入MySQL
    for i in range(0, len(blogs)):
        target_sentence = ''
        headline = blogs[i][2]
        content = blogs[i][3]
        target_sentence += headline
        paragraphs = content.split('\n')
        # 去掉最后一段无用的
        if '截止到今日收盘，各大指数的运行情况如下' in content:
            paragraphs.pop(-1)
        # 处理每一段
        for paragraph in paragraphs:
            if '截止到今日收盘' in paragraph and len(paragraph) < 23:
                continue

            if '子线' in paragraph and '通子' in paragraph and len(paragraph) < 23:
                if '上证指数' in paragraph or '沪深300' in paragraph or '深证综指' in paragraph:
                    if paragraph[-1] == '多':
                        target_sentence += '上涨'
                    elif paragraph[-1] == '空':
                        target_sentence += '下跌'
                    elif paragraph[-1] == '中' or paragraph[-2:] == '中性':
                        target_sentence += '震荡'
                continue

            # 判断段落长度是否小于特定的值
            if len(paragraph) < PARAGRAPH_BOUNDARY:
                target_sentence += paragraph
            else:
                sentences = re.split(r'[。?？]', paragraph)
                if sentences[-1] == '':
                    sentences.pop(-1)
                if len(sentences) <= 2:

                    target_sentence += paragraph[0: int(PARAGRAPH_BOUNDARY / 2)]
                    target_sentence += paragraph[-int(PARAGRAPH_BOUNDARY / 2):]
                elif len(sentences) == 3:
                    target_sentence += sentences[0]
                    target_sentence += sentences[2]
                else:
                    target_sentence += (sentences[0] + sentences[1] + sentences[-2] + sentences[-1])
        # 分词并去除停用词
        seg_list = [x.strip() for x in jieba.cut(target_sentence)
                    if x not in stop_words and x.strip() not in stop_words]
        # 处理同义词
        for index in range(0, len(seg_list)):
            if seg_list[index] in synonyms_dic.keys():
                seg_list[index] = synonyms_dic[seg_list[index]]

        table = 'processed_blogs'
        keys = 'blogger_name, words, created_date, num_words'
        values = ', '.join(['%s'] * 4)
        sql = 'insert into %s (%s) values (%s)' % (table, keys, values)
        try:
            cursor.execute(sql, (blogger_name, ' '.join(seg_list), blogs[i][-1], len(seg_list)))
            db.commit()
        except:
            db.rollback()

    # 测试
    # f = open('./test.txt', 'w', encoding='utf-8', errors='ignore')
    # for word in seg_list:
    #     f.write(word + "\n")
    # f.close()
    #
    # key_words = jieba.analyse.textrank(''.join(content.split('\n')), topK=50, withWeight=True)
    # for word, weight in key_words:
    #     print(word, weight)


if __name__ == '__main__':
    process_original_blogs("余岳桐")

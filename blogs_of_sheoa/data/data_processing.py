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

    # 处理文本并写入MySQL
    for i in range(0, len(blogs)):
        target_sentence = ''
        # 标题
        headline = blogs[i][2]
        if '张中秦：' in headline:
            headline = headline.replace('张中秦：', '')

        content = blogs[i][3]
        target_sentence += headline

        # 去掉无用的段落
        if '郑重声明' in content:
            content = content[0: content.find('郑重声明')]
        if '首山看盘盘中直播' in content:
            content = content[0: content.find('首山看盘盘中直播')]
        if '操作上适当控制仓位，轻仓的可适当增加仓位' in content:
            content = content[0: content.find('操作上适当控制仓位，轻仓的可适当增加仓位')]
        if '的Level2数据显示' in content:
            content = content[0: content.find('的Level2数据显示')]
        if '最新行业板块资金流入比后五名' in content:
            content = content[0: content.find('最新行业板块资金流入比后五名')]
        if '最新一天个股主力净买比例前十名' in content:
            content = content[0: content.find('最新一天个股主力净买比例前十名')]
        if 'Level2数据显示' in content:
            content = content[0: content.find('Level2数据显示')]

        paragraphs = content.split('\n')
        if paragraphs[0] == headline:
            paragraphs.pop(0)
        # 处理每一段
        for paragraph in paragraphs:
            if paragraph.strip() == '':
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


if __name__ == '__main__':
    process_original_blogs("首山")
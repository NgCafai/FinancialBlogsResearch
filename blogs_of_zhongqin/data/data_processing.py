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
                         port=MYSQL_PORT, charset='utf8')
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
        raise Exception('Error when fetching blogs')

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
        if '更多精彩博文点击进入张中秦的博客首页查看' in content:
            content = content[0: content.find('更多精彩博文点击进入张中秦的博客首页查看')]
        if '张中秦寄语' in content:
            content = content[0: content.find('张中秦寄语')]
        if '全方位的市场解读，高效的实时交流' in content:
            content = content[0: content.find('全方位的市场解读，高效的实时交流')]
        if '下载浪客app' in content:
            content = content[0: content.find('下载浪客app')]
        if '个人对大盘大势数据、心得之记录，不构成对任何人的建议。' in content:
            content = content.replace('个人对大盘大势数据、心得之记录，不构成对任何人的建议。', '')
        if '个人对大盘大势数据、心得之记录，不构成对任何人的建议，不接受礼物和打赏。转发、点赞是对秦哥最有力的支持！' in content:
            content = content.replace('个人对大盘大势数据、心得之记录，不构成对任何人的建议，不接受礼物和打赏。转发、点赞是对秦哥最有力的支持！', '')

        paragraphs = content.split('\n')
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
    process_original_blogs("张中秦")
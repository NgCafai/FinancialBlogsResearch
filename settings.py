import datetime

# variables for connecting MySQL
MYSQL_HOST = 'localhost'
MYSQL_DATABASE = 'financial_blogs'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '123456'

# variables for connecting Redis
# Redis数据库地址
REDIS_HOST = '127.0.0.1'
# Redis端口
REDIS_PORT = 6379
# Redis密码，如无填None
REDIS_PASSWORD = None

# 段落长度分界线
PARAGRAPH_BOUNDARY = 150

# 所有博主的名字
blogger_names = ['余岳桐', '张中秦', '首山']

# 训练集开始时间：
train_start = '2009-01-01 00:00:00'
train_end = '2017-10-31 23:59:59'
train_start_date = datetime.date(2009, 1, 1)
train_end_date = datetime.date(2017, 10, 31)
test_start = '2017-11-01 00:00:00'
test_end = '2019-03-12 23:59:59'

# 对未来三天的回报率进行三分类的上下分界线
NEGATIVE_BOUNDARY = -0.0189806007874141
POSITIVE_BOUNDARY = 0.0171055476283131


# 训练模型的配置参数
seq_length = 150  # 序列长度
batch_size = 16  # 每批次的样本数
vocab_size = 5000  # 词汇表大小
num_classes = 2


class ModelConfig(object):
    """
    模型配置参数
    """
    embedding_dim = 32  # 词向量维度
    seq_length = 150  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 210  # 卷积层深度
    kernel_size = [3, 5, 8]  # kernel（或称为filter）的尺寸
    vocab_size = 5000    # 词汇表大小

    hidden_dim = 64  # 全连接层神经元数目

    l2_lambda = 5e-4  # l2正则化的lambda
    dropout_keep_prob = 0.5  # dropout保留比列
    learning_rate = 1e-4  # 学习率

    batch_size = 16  # 每批次的样本数
    num_epochs = 3  # 在所有训练数据上迭代的次数

    print_per_batch = 40  # 每多少个batch输出一次结果
    save_per_batch = 5  # 每多少个batch存入tensorboard

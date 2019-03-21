# -*- coding: utf-8 -*-
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import os
import sys

# 加载自建的package
from Util.utils import *
from model import *
from settings import *


# 最佳模型保存路径
save_dir = './best_model/CNN/best_validation'
# tensorboard保存路径
tensorboard_dir = './tensorboard/CNN'


def train():
    """
    训练模型
    :return:
    """
    start_time = time.time()

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # 使用tensorboar记录运行时的信息
    loss_train_summary = tf.summary.scalar("train/loss", model.loss)
    acc_train_summary = tf.summary.scalar("train/accuracy", model.accuracy)
    merged_train = tf.summary.merge([loss_train_summary, acc_train_summary])
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置Saver，用于保存最佳模型
    saver = tf.train.Saver()
    if not os.path.exists(os.path.split(save_dir)[0]):
        os.makedirs(os.path.split(save_dir)[0])  # 创建保存模型的文件夹：./best_model/CNN

    # 加载经过处理后的训练数据和验证数据
    x_train, y_train, _, _ = get_all_samples(date_returns, word_to_id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        total_batch = 0  # 当前已训练的总批次
        best_acc_on_whole_train = 0.0  # 训练集上的最佳准确率
        last_improved = 0  # 上一次有提升时对应的总批次数
        # 如果超过2000个batch，且在训练集上没有提升，就提前结束训练
        require_improvement = 2000

        flag = False
        for epoch in range(config.num_epochs):
            print('current_epoch(start from 0):', epoch)
            # 用于记录在训练集上的loss和accuracy
            num_samples_trained = 0
            loss_on_whole_train = 0.0
            accuracy_on_whole_train = 0.0

            # 获取每批次训练数据的迭代器
            batch_train = batch_iter(x_train, y_train)
            for x_batch, y_batch in batch_train:
                feed_dict = {model.input_x: x_batch,
                             model.input_y: y_batch,
                             model.keep_prob: config.dropout_keep_prob}

                # 每多少个batch将训练结果用tensorboard记录下来
                if total_batch % config.save_per_batch == 0:
                    feed_dict2 = feed_dict.copy()
                    feed_dict2[model.keep_prob] = 1.0
                    summary = sess.run(merged_train, feed_dict=feed_dict2)
                    writer.add_summary(summary, total_batch)

                # 每多少个batch输出在训练集上的loss和accuracy
                if total_batch % config.print_per_batch == 0:
                    feed_dict2 = feed_dict.copy()
                    feed_dict2[model.keep_prob] = 1.0
                    loss_train, acc_train = sess.run([model.loss, model.accuracy], feed_dict=feed_dict2)

                    msg = 'Total batch:{0:>5}, train loss:{1:>6.2}, train accuracy: {2:>7.2%}'
                    print(msg.format(total_batch, loss_train, acc_train))

                # 使用optimizer优化模型
                _, loss_train, acc_train = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
                loss_on_whole_train += float(loss_train * len(y_batch))
                accuracy_on_whole_train += float(acc_train * len(y_batch))
                num_samples_trained += len(y_batch)

                total_batch += 1

            accuracy_on_whole_train = accuracy_on_whole_train / float(num_samples_trained)
            loss_on_whole_train = loss_on_whole_train / float(num_samples_trained)
            msg = 'Epoch:{0:>5}, loss on whole train:{1:>6.2}, accuracy on whole train: {2:>7.2%}'
            print(msg.format(epoch, loss_on_whole_train, accuracy_on_whole_train))
            # 保存在训练集上的最佳模型
            if accuracy_on_whole_train > best_acc_on_whole_train:
                best_acc_on_whole_train = accuracy_on_whole_train
                saver.save(sess=sess, save_path=save_dir)


    # 输出训练时间
    end_time = time.time()
    print('total train time:', timedelta(seconds=int(round(end_time - start_time))))


# def test():
#     x_test, y_test = process_file(processed_test_dir, word_to_id, category_to_id, config.seq_length)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         saver.restore(sess=sess, save_path=save_dir)
#
#         feed_dict = {
#             model.input_x: x_test,
#             model.input_y: y_test,
#             model.keep_prob: 1.0
#         }
#         loss_test, acc_test = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
#         msg = 'test loss: {0:>6.2}, test accuracy: {1:>7.2%}'
#         print(msg.format(loss_test, acc_test))


if __name__ == '__main__':
    config = ModelConfig()  # 模型的各项参数
    words, word_to_id = read_vocab('./Util/')  # 读取字典
    date_returns: dict = get_date_returns('./Util/')  # 每个日期接下来三个交易日的return
    config.vocab_size = len(words)
    # 建立CNN模型
    model = Model(config)

    train_or_test = 'train'
    if train_or_test == 'train':
        train()
    else:
        pass
        # test()

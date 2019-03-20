from settings import *
from jqdatasdk import *
import csv
import pickle
from matplotlib import pyplot as plt
import math
import datetime
from settings import *


def get_data():
    # create the csv file in which we are going to write the data
    csv_file = open('./stock_data.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['date', 'close', 'high', 'low', 'next_three_day_high',
                     'next_three_day_low', 'next_three_day_return'])

    # get stock price from joinquant
    auth('13286176580', 'manu5f3acturi')
    df = get_price('000300.XSHG', '2008-01-01', '2019-03-15', fields=['close', 'high', 'low'])
    # a sample element in date: datetime.date(2019, 3, 1)
    # when printed: 2019-03-01
    date = [element.date() for element in df.index.tolist()]

    close = df['close'].tolist()
    high = df['high'].tolist()
    low = df['low'].tolist()

    if len(close) != len(date):
        raise Exception('Error')

    length = len(close)
    next_three_day_returns = []

    for i in range(0, length - 3):
        new_row = [date[i], close[i], high[i], low[i]]
        next_three_day_high = max(high[(i + 1): (i + 4)])
        next_three_day_low = min(low[(i + 1): (i + 4)])
        max_return = (next_three_day_high - close[i]) / close[i]
        min_return = (next_three_day_low - close[i]) / close[i]
        next_three_day_return = max_return if abs(max_return) > abs(min_return) else min_return
        new_row.append(next_three_day_high)
        new_row.append(next_three_day_low)
        new_row.append(next_three_day_return)
        next_three_day_returns.append(next_three_day_return)

        writer.writerow(new_row)

    with open('./next_three_day_returns.pk', 'wb') as f:
        pickle.dump(next_three_day_returns, f)
    with open('date.pk', 'wb') as f:
        pickle.dump(date[0: length - 3], f)
    csv_file.close()


if __name__ == '__main__':
    # get_data()
    with open('./next_three_day_returns.pk', 'rb') as f:
        next_three_day_returns = pickle.load(f)

    with open('date.pk', 'rb') as f:
        date = pickle.load(f)
    print(type(date[0]))
    # print(date)
    print(len(next_three_day_returns))
    # print(next_three_day_returns)

    # dic = {k: v for k, v in zip(date, next_three_day_returns)
    #        if k.__ge__(train_start_date) and k.__le__(train_end_date)}
    # sorted_dic = sorted(dic.items(), key=lambda d: d[1])
    # with open('./sorted_returns.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for key, value in sorted_dic:
    #         writer.writerow([key, value])




    # sorted_returns = next_three_day_returns.copy()
    # sorted_returns.sort()
    # plt.hist(next_three_day_returns)
    # plt.show()
    # plt.boxplot(next_three_day_returns)
    # plt.show()
    # plt.plot(next_three_day_returns)
    # plt.show()
    # plt.plot(sorted_returns)
    # plt.show()











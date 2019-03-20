import pymysql
from settings import *
from jqdatasdk import *

if __name__ == '__main__':
    db = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
                         port=MYSQL_PORT)
    cursor = db.cursor()

    sql = 'select * from blogs where id = 405'
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print('Error')

    auth('13286176580', 'manu5f3acturi')
    df = get_price('000001.XSHG', '2019-03-01', '2019-03-12', fields=['close', 'pre_close'])
    df['return'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['return'].tolist()

    print(df)
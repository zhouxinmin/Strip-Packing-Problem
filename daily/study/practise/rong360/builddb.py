# coding:utf-8
import MySQLdb

conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='123',
        db='rong360',
        )
cur = conn.cursor()
useinfo = open('H:/Datamining/rong360/credit/train/user_info_train.txt')
for line in useinfo:
    linelist = line.split(',')
    # sqli = "insert into useinfo values(%s,%s,%s,%s,%s,%s)"
    # cur.execute(sqli, ('', '1', '2', '4', '4', '2'))
    cur.execute("INSERT INTO useinfo (useid, sex, profession, edu, marrige, resident)VALUES(%s,%s,%s,%s,%s,%s)",
                [linelist[0], linelist[1], linelist[2], linelist[3], linelist[4], linelist[5]])

cur.close()         # 关闭游标
conn.commit()       # 提交事物，在向数据库插入一条数据时必须要有这个方法，否则数据不会被真正的插入
conn.close()        # 关闭数据库连接
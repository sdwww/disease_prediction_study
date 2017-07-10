import os
import datetime

from openpyxl import load_workbook

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
import cx_Oracle as db
import numpy as np
np.random.seed(1234)

# 连接数据库
def connect():
    con = db.connect('MH3', '123456', '127.0.0.1:1521/ORCL')
    return con


# 执行select语句
def getSQL(sql, cursor):
    cursor.execute(sql)
    result = cursor.fetchall()
    content = []
    for row in result:
        if len(row) == 1:
            content.append(row[0])
        else:
            content.append(list(row))
    return content


# 执行更新操作
def exeSQL(sql, cursor, con):
    cursor.execute(sql)
    con.commit()
    return 1


# 获取数据库的部分内容
def getAllInfo(cursor, spm_list, jbmc_list):
    all_info = []
    for i in range(49):
        sql = 'SELECT * from MH3.DATA_ANALYSIS where rownum <=' + str(1000000 + i * 1000000) + \
              ' MINUS SELECT * from DATA_ANALYSIS where rownum <= ' + str(i * 1000000)
        batch_info = getSQL(sql, cursor)
        for i in batch_info:
            if i[4] in jbmc_list and i[5] in spm_list:
                all_info.append(i)
    return all_info

import time

import numpy as np

import DBOptions

patient_num = 28753
visit_num = 40
disease_num = 1236
drug_num = 1096
disease_category_num = 588
split_rate = 0.85


def create_dataset(cursor):
    # 创建个人信息数据集
    sql = 'select GRBH_INDEX,GENDER,AGE from DATA_ANALYSIS_INFO'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_info = np.zeros((patient_num, 4), dtype='int16')
    print('信息数据集大小：', dataset_info.shape)
    for i in all_info:
        if i[1]:
            dataset_info[int(i[0]), 0] = int(i[1])
        if not i[1]:
            dataset_info[int(i[0]), 1] = 1
        if i[2]:
            dataset_info[int(i[0]), 2] = str(i[2])
        if not i[2]:
            dataset_info[int(i[0]), 3] = 1

    # 创建诊断数据集
    sql = 'select grbh_index,xh_index,jbbm_index from DATA_ANALYSIS_JBBM where XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_disease = np.zeros((patient_num, visit_num, disease_num), dtype='int16')
    print('诊断数据集大小：', dataset_disease.shape)
    for i in all_info:
        if i[1] < visit_num:
            dataset_disease[i[0], -i[1], i[2]] = 1

    # 创建药品数据集
    sql = 'select grbh_index,xh_index,DRUG_INDEX,ZJEITEM from DATA_ANALYSIS_DRUG,DATA_ANALYSIS_JBBM where ' \
          'DATA_ANALYSIS_JBBM.XH =DATA_ANALYSIS_DRUG.XH AND DATA_ANALYSIS_JBBM.XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_drug = np.zeros((patient_num, visit_num, drug_num), dtype='int16')
    print('药品数据集大小：', dataset_drug.shape)
    for i in all_info:
        if i[1] < visit_num:
            dataset_drug[i[0], -i[1], i[2]] += i[3]

    time_list = ['2014-10-29', '2015-01-29', '2015-04-29', '2015-07-29']
    for i in range(len(time_list)):
        sql = "select grbh_index,jbmc_categ_index from DATA_ANALYSIS_JBBM where XH_INDEX=0 " \
              "and ZYRQ <= to_date('" + time_list[i] + " 00:00:00','yyyy-mm-dd hh24:mi:ss')"
        all_info = DBOptions.get_sql(sql, cursor=cursor)
        label_disease = np.zeros((patient_num, disease_category_num), dtype='int16')
        label_probability = np.zeros((patient_num, 1), dtype='int16')
        print('诊断标签大小：', label_disease.shape)
        for item in all_info:
            label_disease[item[0], item[1]] = 1
            label_probability[item[0], 0] = 1

        np.savez_compressed('./dataset/dataset_' + str(12 // len(time_list) * (i + 1)) + 'month.npz',
                            dataset_info=dataset_info,
                            dataset_disease=dataset_disease, dataset_drug=dataset_drug,
                            label_disease=label_disease, label_probability=label_probability)


if __name__ == "__main__":
    start = time.clock()
    connect = DBOptions.connect_db()
    cursor = connect.cursor()
    create_dataset(cursor=cursor)
    cursor.close()
    connect.close()
    print(time.clock() - start)

# -*- coding:utf8 -*-
# 从爬取数据文件中更新通过Tenseract OCR识别的值

import pandas as pd
import pymongo
import json

f = open('/Users/ethan/MyCodes/baidu_tensor/index_20170707.txt')

contents = f.read().split('\n')[:-1]

new_list = []

for line in contents:
    new_list.append(json.loads(line))

df = pd.DataFrame(new_list)

db = pymongo.MongoClient('localhost', 27017).test

for ix, item in df.iterrows():
    db.files.update_one({'name': item['keyword'], 'type':'total'},{"$set": {"OCR":item['totalIndex']}})
    db.files.update_one({'name': item['keyword'], 'type':'search'},{"$set": {"OCR":item['searchIndex']}})

#item = df.loc[0]

#f = db.files.find_one({'name': item['keyword'], 'type':'total'})


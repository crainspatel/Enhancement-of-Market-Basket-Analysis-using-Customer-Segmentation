# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 00:26:57 2021

@author: madpa
"""

#%%
import pandas as pd
import numpy as np

import pyspark
from pyspark import SparkContext
#%%
from pyspark.mllib.fpm import FPGrowth

#%%
from pyspark.ml.fpm import FPGrowth

#%%
SparkContext.setSystemProperty('spark.executor.memory', '16g'),('spark.driver.memory', '10g'),('spark.driver.maxResultSize', '10g') ## adjust the required memory to your computational resource!
sc = pyspark.SparkContext()

from pyspark.sql.session import SparkSession
spark = SparkSession(sc)
from pyspark.sql.functions import split

#%%
aisles_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/aisles.csv')
departments_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/departments.csv')
orders_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/orders.csv')
products_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/products.csv')
order_product_prior_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/order_products__prior.csv')
order_product_train_df = pd.read_csv('D:/NEU/DS 5230/Project/Instacart/Data/order_products__train.csv')

#%%
baskets = {}
#%%
import pickle
order_product_prior_refined_df=pickle.load( open( "D:/NEU/DS 5230/Project/Instacart/Data/out4.p", "rb" ) )

order_list = order_product_prior_refined_df['order_id'].unique()
order_list

order_product_prior_refined_df['index'] = range(0, len(order_product_prior_refined_df))

order_product_prior_refined_df = order_product_prior_refined_df.set_index('index')

g = order_product_prior_refined_df['order_id']
#%%
order_product_prior_refined_df = order_product_prior_refined_df.applymap(str)

#%%
d = order_product_prior_refined_df.groupby(g, sort=False).first()
d['product_id'] = order_product_prior_refined_df['product_id'].dropna().groupby(g).agg(','.join)
d = d.reset_index(drop=True)

#%%
np.savetxt(r'D:/NEU/DS 5230/Project/Instacart/Data/product_baskets.txt', d['product_id'].values, fmt='%s')

#%%
data = (spark.read.text("D:/NEU/DS 5230/Project/Instacart/Data/product_baskets.txt").select(split("value", "\s+").alias("items")))
#%%  Assignment FP Growth
rdd2 = sc.textFile('D:/NEU/DS 5230/Project/Instacart/Data/product_baskets.txt')
itemscol2 = rdd2.map(lambda line: line.split(','))
#%% 
model = FPGrowth.train(itemscol2,minSupport=1e-3,numPartitions=1)

#%%
fpGrowth = FPGrowth(itemsCol="items", minSupport=1e-3, minConfidence=0.6)
model = fpGrowth.fit(data)

#%%

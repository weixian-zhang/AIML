#!/usr/bin/env python
# coding: utf-8

# ## Kaggle - Regression Practice for DS Online FT-011121 Cohort
# 
# New notebook

# In[3]:


# Welcome to your new notebook
# Type here in the cell editor to add code!

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pyspark.pandas as pypd
import pickle


# ### Data Loading
# 
# Split training data for validation

# In[4]:


df_train = spark.read.csv(
    'abfss://bb32fd16-8b44-43f1-9500-dcaca93186a1@msit-onelake.dfs.fabric.microsoft.com/e4197549-0c6a-45c0-bdf5-de12743d2825/Files/kaggle-regression/train.csv',
    header=True,
    inferSchema=True)

display(df_train.limit(5))


# ### Data Cleaning

# keep useful features only

# In[5]:


from pyspark.sql.types import IntegerType, FloatType

col_to_keep = ['Yr.Sold', 
'Lot.Frontage', 
'Lot.Area', 
'Gr.Liv.Area', 
'Full.Bath', 
'Half.Bath', 
'Neighborhood',
'SalesPrice']

for c in df_train.columns:
    if c not in col_to_keep:
        df_train = df_train.drop(c)

df_train = df_train.withColumnRenamed('Yr.Sold', 'YrSold')
df_train = df_train.withColumnRenamed('Lot.Frontage', 'LotFrontage')
df_train = df_train.withColumnRenamed('Lot.Area', 'LotArea')
df_train = df_train.withColumnRenamed('Gr.Liv.Area', 'GrLivArea')
df_train = df_train.withColumnRenamed('Full.Bath', 'FullBath')
df_train = df_train.withColumnRenamed('Half.Bath', 'HalfBath')

df_train = df_train.withColumn('LotFrontage', df_train['LotFrontage'].cast(IntegerType()))
df_train = df_train.withColumn('LotArea', df_train['LotArea'].cast(IntegerType()))
df_train = df_train.withColumn('GrLivArea', df_train['GrLivArea'].cast(IntegerType()))
df_train = df_train.withColumn('YrSold', df_train['YrSold'].cast(IntegerType()))
df_train = df_train.withColumn('SalesPrice', df_train['SalesPrice'].cast(FloatType()))
df_train = df_train.withColumn('FullBath', df_train['FullBath'].cast(IntegerType()))
df_train = df_train.withColumn('HalfBath', df_train['HalfBath'].cast(IntegerType()))

df_train.printSchema()


# ### One Hot Encoding - Neighbourhood

# In[28]:


from pyspark.sql.types import IntegerType

pandas_df_train = df_train.toPandas()

neighbourhood_numpy = pandas_df_train['Neighborhood'].to_numpy()

neighbourhood_numpy = neighbourhood_numpy.reshape(-1,1)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.set_output(transform='pandas')

# fit then transform
# https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial
unique_neighbourhoods = np.array(pandas_df_train['Neighborhood'].unique()).reshape(-1, 1)

ohe.fit(unique_neighbourhoods)

ohe_transformed = ohe.transform(neighbourhood_numpy)

df_ohe_transformed = spark.createDataFrame(ohe_transformed).toPandas()

final_ohe = pd.concat([pandas_df_train, df_ohe_transformed], axis=1)#.drop(columns=['Neighborhood'])

df_train_ohe = spark.createDataFrame(final_ohe)

df_train_ohe = df_train_ohe.drop('Neighborhood', 'x0_Veenker')

display(df_train_ohe.head(10))


# ### fill Nan and Null with SimpleImputer

# In[29]:


from sklearn.impute import SimpleImputer
import pyspark.sql.functions as F

df_train_ohe.toPandas().info()

# df = df_train_ohe.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for c in df_train_ohe.columns])

# display(df) # row count of nan and null

df_train_ohe_pandas = df_train_ohe.toPandas()

imp_mean = SimpleImputer(strategy='median')

df_train_ohe_pandas['LotFrontage'] = imp_mean.fit_transform(df_train_ohe_pandas['LotFrontage'].to_numpy().reshape(-1, 1)).flatten()

df_train_ohe = spark.createDataFrame(df_train_ohe_pandas)

df_train_ohe.toPandas().info()

# display(df_train_ohe.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for c in df_train_ohe.columns]))


# ### Feature Scaling - TODO

# In[44]:


df_train_ohe.head(20)


# ### Data Splitting - Train, Test & Validation

# In[8]:


from pyspark.sql.types import IntegerType

pandas_df_train_ohe = df_train_ohe.toPandas()

X = pandas_df_train_ohe.drop(columns=['SalesPrice'])
y = pandas_df_train_ohe['SalesPrice'].to_numpy()

# y_train, y_test = df_train_ohe.select('SalesPrice').randomSplit([0.7,0.3], seed=3000)
# X_train, X_test = df_train_ohe.drop('SalesPrice').randomSplit([0.7, 0.3], seed=3000)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size= 0.2)

# LotFrontage = [x['LotFrontage'] for x in X_train.select('LotFrontage').collect()] #[x['LotFrontage'] for x in X_train]
# LotArea = [x['LotArea'] for x in X_train]
# GrLivArea = [x['GrLivArea'] for x in X_train]
# x0_Blueste = [x['x0_Blueste'] for x in X_train]
# x0_Blmngtn = [x['x0_Blmngtn'] for x in X_train]

# y_train_nums = [x['SalesPrice'] for x in y_train.select('SalesPrice').collect()]

# plt.scatter(LotFrontage, y_train_nums)

#fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(5, 1)

#axis1.scatter(np.array(LotFrontage), np.array(y_train))
# axis2.scatter(LotArea, np.array(y_train))
# axis3.scatter(GrLivArea, np.array(y_train))
# axis4.scatter(x0_Blueste, np.array(y_train))
# axis5.scatter(x0_Blmngtn, np.array(y_train))

# # display(X_train)


# ### Prepare Spark Dataframe for Linear Regression

# In[31]:


# a = np.array(neighbourhood_numpy[3])

# ohe.transform(a.reshape(-1,1))
# from pyspark.ml.feature import VectorAssembler


# X_train_df = spark.createDataFrame(X_train)

# cols = [c for c in X_train_df.columns]

# assembler = VectorAssembler(inputCols=cols, outputCol='features')

# df_features_assembled = assembler.transform(X_train_df)

# X_vector_arr = [row[0] for row in df_features_assembled.select('features').collect()]



# 

# ### Model Training

# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    PolynomialFeatures(degree=1, include_bias=False), 
    LinearRegression())

pipeline.fit(X_train, y_train)

y_tests_predictions = pipeline.predict(X_test)

y_train_predictions = pipeline.predict(X_train)






# ### Model Evaluation  
# 
# * R^2 a.k.a R-squared
# * mean absolute error
# * mean squared error

# In[46]:


# https://www.youtube.com/watch?v=VzbowFygnVw
# multiple linear regression - https://www.youtube.com/watch?v=wH_ezgftiy0&t=208s
# pandas - iloc vs loc - https://www.youtube.com/watch?v=lJDtzZsmF0g

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

score = cross_val_score(pipeline, X_train, y_train)

(
score.mean(), score.std(),
r2_score(y_test, y_tests_predictions), 
mean_squared_error(y_test, y_tests_predictions),
mean_absolute_error(y_test, y_tests_predictions),
)

# X_train

# p = pipeline.predict(np.array(X_test.iloc[[5]]))


# plt.plot(X_test.iloc[:,2], predictions, color='g')
# fig, (axis1, axis2) = plt.subplots(1,2)
# axis1.scatter(y_test, x_tests_predictions)
# axis2.scatter(y_train, x_train_predictions)


# 

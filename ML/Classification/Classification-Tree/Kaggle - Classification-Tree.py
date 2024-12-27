# Databricks notebook source
# MAGIC %md
# MAGIC # [Kaggle Decision Tree Classifier](https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial)

# COMMAND ----------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import graphviz

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load Data

# COMMAND ----------



sdf = spark.sql('select * from car_evaluation')

df = sdf.toPandas()

df['class'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Analysis

# COMMAND ----------

df.iloc[:10, :2]

# COMMAND ----------

print(df.shape)

df.info()

# COMMAND ----------

for c in df.columns:
    print(df[c].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Declare feature vector and target variable 

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df.drop(['class'], axis=1)

Y = df['class']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(x_train.shape, x_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Engineering

# COMMAND ----------

x_train['lug_boot'].unique()

# COMMAND ----------

df['safety'].unique()

# COMMAND ----------

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

x_train = encoder.fit_transform(x_train)

x_test = encoder.transform(x_test)

# COMMAND ----------

x_test

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Training Model - Decision Tree Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### with Gini Index

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

dtc_gini = DecisionTreeClassifier()

dtc_gini.fit(x_train, y_train)

y_pred_gini = dtc_gini.predict(x_test)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## with Entopy & Information Gain 

# COMMAND ----------


dtc_entropy = DecisionTreeClassifier(criterion='entropy')

dtc_entropy.fit(x_train, y_train)

y_pred_entropy = dtc_entropy.predict(x_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classifier Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation between Gini Index and Entropy

# COMMAND ----------

from sklearn.model_selection import cross_val_score


gini_score = cross_val_score(dtc_gini, x_train, y_train)

entropy_score = cross_val_score(dtc_entropy, x_train, y_train)

print(gini_score)
print(entropy_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ### training score

# COMMAND ----------

from sklearn.metrics import accuracy_score

print(f'training score: {dtc_gini.score(x_train, y_train)}')

print(f'testing score: {dtc_gini.score(x_test, y_test)}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### plot sklearn.tree

# COMMAND ----------


from sklearn import tree

tree.plot_tree(dtc_gini.fit(x_train, y_train))




# COMMAND ----------

dot_data = tree.export_graphviz(gini_score, out_file=None, 
                              feature_names=x_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 



# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix


confusion_matrix(y_test, y_pred_gini)

# COMMAND ----------

pd.DataFrame(y_test)['class'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_gini))

# COMMAND ----------



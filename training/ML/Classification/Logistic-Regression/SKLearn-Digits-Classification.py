# Databricks notebook source
from scipy.stats import norm

x1 = -1
x2 = 1
std = 10
mean = -5
a = norm.cdf(x1, mean, std)
b = norm.cdf(x2, mean, std)

print(a)
print(b)
print(b-a)

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

x =np.arange(1,10)
y = (x ** 2) + (12 * x) +5# square term

print(x)
print(y)
plt.plot(x,y)
plt.show()





# COMMAND ----------

# MAGIC %md
# MAGIC # SKLearn Digits Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import seaborn as sns


# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

digits = load_digits()






# COMMAND ----------

digits.data.shape

# COMMAND ----------

plt.figure(figsize=(20, 4))

for image, label in zip(digits.images[:1], digits.target[:1]):
    print(image)
    plt.imshow(image, cmap='gray')

# COMMAND ----------

digits.target[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split Dataset

# COMMAND ----------

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

log_regression = LogisticRegression()

log_regression.fit(x_train, y_train)

# COMMAND ----------

plt.imshow(x_test[14].reshape(8, 8))

# COMMAND ----------



log_regression.predict(x_test[14].reshape(1, -1))

# COMMAND ----------

predictions = log_regression.predict(x_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Measuring Model Performance

# COMMAND ----------

accuracy_score = log_regression.score(x_test, y_test)
print(accuracy_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

cm = confusion_matrix(y_test, predictions)
print(cm)

# COMMAND ----------


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title = f'Accuracy Score = {accuracy_score}'

# COMMAND ----------

# MAGIC %md
# MAGIC # MNIST Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load MNIST Data

# COMMAND ----------

from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# COMMAND ----------

print(X[0].reshape(28, 28))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Data

# COMMAND ----------

plt.figure(figsize=(10,10))
plt.imshow(X[0].reshape(28, 28))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Split Dataset Train & Test

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_img, test_img, train_lbl, test_lbl = train_test_split(X, y, test_size=1/7.0, random_state=42)

# COMMAND ----------



plt.figure(figsize=(20,10))

for index, (img,label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

log_regression_2 = LogisticRegression(solver='lbfgs')

log_regression_2.fit(train_img, train_lbl)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prediction

# COMMAND ----------



predictions = log_regression_2(test_img, test_lbl)

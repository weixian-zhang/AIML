# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Equation of Least Squares Regression Line - Correlation

# COMMAND ----------

x= np.array([1,2,2,3])
y= np.array([1,2,3,6])

y_mean = np.mean(y, axis=0)
x_mean = np.mean(x, axis=0)

correlation = np.corrcoef(x,y)[0,1]
x_stdev = np.std(x)
y_stdev = np.std(y)

# y = mx + b

# manual calc using below formula to find m and b which is the slope and y-interept or coefficients of y = mx + b
m = correlation * (y_stdev / x_stdev) # 2.5
b = y_mean - (m * x_mean)

# use numpy polyfit instead of manual calc
coefficients, least_squares, _, _, _ = np.polyfit(x,y,1, full=True)
numpy_m = coefficients[0]
numpy_b = coefficients[1]

print(f'm={round(m,2)}')
print(f'b={round(b,2)}')
print(f'numpy m= {numpy_m}')
print(f'numpy b= {numpy_b}')

y_predict = m * x + b

# sum of squares
ss = (y_predict - y) ** 2 
# m = 2.5
# b = -2

# formula
# m = corelation * standard deviation sx / standard deviation ​sy​​
# b = y_mean ​− m * x_mean

print(f'corelation: {correlation}')
print(f'x_stdev: {x_stdev}')
print(y_stdev)
print(ss)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get Equation of Regression line - another formula
# MAGIC
# MAGIC ![](/Workspace/Users/admin@mngenvmcap049172.onmicrosoft.com/AIML/ML/find equation for regression line.png)

# COMMAND ----------


# n: Number of data points.
# ΣxΣx: Sum of all xx values.
# ΣyΣy: Sum of all yy values.
# ΣxyΣxy: Sum of the product of each pair of xx and yy.
# Σx2Σx2: Sum of squares of each xx.

x= np.array([1,2,2,3])
y= np.array([1,2,3,6])

n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)

coefficients, least_squares, _, _, _ = np.polyfit(x,y,1, full=True)

print(f'coefficients: {coefficients}')
print(f'least_squares: {least_squares}')



# COMMAND ----------

# MAGIC %md
# MAGIC ## Polynomial Function & Regression

# COMMAND ----------

x = np.arange(1,8)

y = np.array([53807, 55843, 55209,56415,56811, 57666,58803])


plt.x_label='year from 1965'
plt.y_label = 'income'
plt.scatter(x,y, c='black', s=100)

# get coefficient of poly func
coeff = np.polyfit(x,y,1)

# create polynomial function
polynomial_func = np.poly1d(coeff)
print(f'coefficients: {coeff}')
print(f'polynomial function: {polynomial_func}')

xx = np.linspace(1,len(x),10)
yy = polynomial_func(x)

# print(xx)
# print(yy)

# plt.plot(x,yy,c='blue')

# COMMAND ----------



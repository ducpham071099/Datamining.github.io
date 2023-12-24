# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:42:04 2023

@author: hoang
"""
### Tra Mai Hoang, Anh Duc Ngoc Pham 
### MSIS 672
### Assignment 1 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# 1. Read the data and create a data frame for the data set 
import pandas as pd
cereal_df = pd.read_csv('Cereals.csv')
# 2. List all attributes’ names and data types
    # Show all attributes' names
cereal_df.columns 
    # Show all data types
datatype = dict(cereal_df.dtypes)
datatype
        #or
datatype = cereal_df.dtypes
datatype

# 3. List the dimension of the data frame (the number of rows and the number of columns)
cereal_df.shape
 # 77 rows, 16 columns
 
# 4. List the summary statistics for all numerical types of columns. 

result_df= pd.concat([cereal_df.iloc[0:78,3:17]], axis =1)  #subset the numerical types column only
result_df
result_df.describe()

#or

cereal_df.iloc[:,3:16].describe()

#or

cereal_numerical_df = cereal_df.drop(columns=['name', 'mfr','type'])
cereal_numerical_df .describe()

# 5. Show the distribution of “calories” using a graph
import matplotlib.pylab as plt

ax = cereal_df.calories.hist()
ax.set_xlabel('calories')
ax.set_ylabel('count')

#6. Please show the distribution of “mfr” using a  graph
ax = cereal_df.mfr.hist()
ax.set_xlabel('mfr')
ax.set_ylabel('count')

# 7. Examine the relationship between ‘calories” and “rating” using a scatter plot (calories as x and rating as y)
cereal_df.plot.scatter(x='calories', y='rating', legend=False)

# 8. Build a linear regression model to predict “rating” (rating is the outcome variable)
    # 8.1 Please drop the categorical variable  'name'. 
remove_name = 'name'
new_df = cereal_df.drop(remove_name, axis=1)
new_df

    # 8.2 Check the remaining columns’ types and create dummy variables for non-numerical columns (pandas.get_dummies ()). Please make sure to drop the first dummy variable’s value (drop_first=True) (hint: the sample code is in chapter 2)
new_df.dtypes

new_df.columns
new_df = pd.get_dummies(new_df, prefix_sep='_', drop_first=True)
new_df.columns

    # 8.3 Check missing values. If there are missing values, please replace missing values with the column’s median value (chapter 2).  Please make sure there are no missing values.

# Find out which columns have missing values
new_df.count()
# -> carbo, sugars and potass 

median_carbo = new_df['carbo'].median() #calculate median value
median_carbo
new_df.carbo = new_df.carbo.fillna(value=median_carbo) # replace the missing values with the median value
print('Number of rows with valid carbo values after filling NA values: ', new_df['carbo'].count())

median_sugars = new_df['sugars'].median() #calculate median value
median_sugars
new_df.sugars = new_df.sugars.fillna(value=median_sugars)
print('Number of rows with valid sugars values after filling NA values: ', new_df['sugars'].count())

median_potass = new_df['potass'].median() #calculate median value
median_potass
new_df.potass = new_df.potass.fillna(value=median_potass)
print('Number of rows with valid potass values after filling NA values: ', new_df['potass'].count())

    # 8.4 Split the data into training (70%) and validation (30%)

trainData, validData = train_test_split(new_df, test_size=0.30, random_state=1)
print('Training : ', trainData.shape)
print('Validation : ', validData.shape)

    #8.5 Build the regression model using LinearRegression (). Display the model (intercept and coefficients)

from dmba import regressionSummary#, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import AIC_score #, BIC_score, adjusted_r2_score
import statsmodels.formula.api as sm

# create a list containing predictors' name
predictors = ['calories', 'protein', 'fat','sodium','fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups'] 
print(predictors)

# define outcome/target variable
outcome = 'rating'
print(outcome)

new_df[predictors].dtypes 

x = pd.get_dummies(new_df[predictors], drop_first=True) 

y = new_df[outcome]

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.30,random_state=1) 
train_x.head()

# Create LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)

#print coeficients
print(pd.DataFrame({'predictor': x.columns, 'coefficient': model.coef_}))

model.intercept_

     #8.6 Please assess the model’s performance on training and validation data (regressionSummary())
# print performance measures (training data)
regressionSummary(train_y, model.predict(train_x))






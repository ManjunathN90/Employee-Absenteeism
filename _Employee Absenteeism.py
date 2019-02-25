
#Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from scipy.stats import chi2_contingency


# Load the dataset


df = pd.read_excel('Absenteeism_at_work_Project.xls')



#  Intial Analysis

df.info()

df.shape


df.describe()




############### Missing Value Analysis ######################



missing_val = pd.DataFrame(df.isnull().sum())

missing_val = missing_val.reset_index()

missing_val = missing_val.rename(columns = {'index': 'Variables', 0:'Missing_Percentage'})

missing_val['Missing_Percentage'] = (missing_val['Missing_Percentage']/ len(df))*100

missing_val =  missing_val.sort_values('Missing_Percentage', ascending=False).reset_index(drop = True)


#Impute Method
#df['Absenteeism time in hours'][28]
#Actual Value = 8
#mean = 6.97
#median = 3.0
#KNN = 7.11



#Knn_imputation for missing values
df = pd.DataFrame(KNN(k=3).complete(df), columns=df.columns)



#Round of the numeric values

for i in df.columns:
    df[i] = df[i].round()


########### Exploratory Data Analysis ############


cnames = ['Transportation expense', 'Distance from Residence to Work','Service time', 'Age', 'Work load Average/day '
          ,'Hit target','Weight','Height','Body mass index','Absenteeism time in hours']


#Distribution plot with kernel density for all the predictors
for i in cnames:
    k = sns.distplot(df[i])
    plt.show()
  


#Graphs to analyse the data

for i in df.columns:
    sns.barplot(x = i, y = "Absenteeism time in hours", data = df)
    plt.tight_layout()
    plt.show()



for i in df.columns:
    plt.figure(figsize = (15,6))
    sns.barplot(x = i, y = "Absenteeism time in hours", data = df)
    plt.tight_layout()



plt.figure(figsize = (18,6))
sns.barplot(x='Age', y='Absenteeism time in hours', hue = 'Service time',data=df)
plt.tight_layout()



################ Outlier Analysis #########################


for i in cnames:
    plt.boxplot(df[i])
    plt.xlabel(i)
    plt.show()
  


# Continious variables


cnames = ['Transportation expense', 'Distance from Residence to Work','Service time', 'Age', 'Work load Average/day '
          ,'Hit target','Weight','Height','Body mass index','Absenteeism time in hours']



#With outliers
for i in cnames:
    q75, q25 = np.percentile(df.loc[:,i], [75 ,25])
    iqr = q75 - q25

    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    print(minimum)
    print(maximum)
   
    print(len(df[df.loc[:,i] < minimum]))
    print(len(df[df.loc[:,i] > maximum]))
    
    df[i][df.loc[:,i] < minimum] = np.nan
    df[i][df.loc[:,i] > maximum] = np.nan

#Impute NA with KNN_Imputation
df = pd.DataFrame(KNN(k = 3).complete(df), columns = df.columns)


############# Feature Selection ###########


df_corr = df.loc[:,cnames]

corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
#plt.savefig("heatmap.png")



catnames = ['ID','Reason for absence', 'Month of absence','Age', 'Day of the week','Seasons','Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet',]



for i in catnames:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(df['Absenteeism time in hours'], df[i]))
    print(p)



# Dimensionality Reduction
df = df.drop(['Body mass index'], axis = 1)



df.shape



################ Feature Scaling #################


for i in df.columns:
    print(i)
    df[i] = (df[i] - np.min(df[i])) / (np.max(df[i]) - np.min(df[i]))




################## Model Development ##############



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error



#split the dataset into train and test dataset 
X = df.iloc[:,0:19]
y = df.iloc[:,19]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


####Decision Tree Regressor


DT_reg = DecisionTreeRegressor(max_depth=2)

DT_reg.fit(X_train,y_train)

#predict test cases
Prediction_DT = DT_reg.predict(X_test)


# Calculate RMSE and MAE

rmse_for_test = np.sqrt(mean_squared_error(y_test,Prediction_DT))

MAE = mean_absolute_error(y_test, Prediction_DT)


##Error metrics for Decision Tree
#MAE =  0.13
#RMSE = 0.18



#### Random Forest


RF_Reg = RandomForestRegressor()

RF_Reg.fit(X_train, y_train)

#predict the test cases
Prediction_RF = RF_Reg.predict(X_test)


rmse_for_test_RF = np.sqrt(mean_squared_error(y_test,Prediction_RF))


MAE = mean_absolute_error(y_test, Prediction_RF)



##Error metrics for Random Forest
#MAE =  0.12
#RMSE = 0.17


#### Linear Regression


import statsmodels.api as sm

model = sm.OLS(y_train, X_train).fit()

model.summary()

Prediction_LR = model.predict(X_test)


rmse_for_test_LR = np.sqrt(mean_squared_error(y_test,Prediction_LR))


MAE = mean_absolute_error(y_test, Prediction_LR)



##Error metrics for Random Forest
#MAE =  0.14
#RMSE = 0.18




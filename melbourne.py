import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_validate

df = pd.read_csv('Melbourne_housing_FULL.csv')

#print (df)

#print(metrics.SCORERS.keys())


regionname = pd.get_dummies(df['Regionname'],drop_first=True)
df = pd.merge(df, regionname, left_index=True, right_index=True)
df.drop('Regionname', axis=1, inplace=True)

suburb = pd.get_dummies(df['Suburb'],drop_first=True)
df = pd.merge(df, suburb, left_index=True, right_index=True)
df.drop('Suburb', axis=1, inplace=True)

council = pd.get_dummies(df['CouncilArea'],drop_first=True)
df = pd.merge(df, council, left_index=True, right_index=True)
df.drop('CouncilArea', axis=1, inplace=True)

method = pd.get_dummies(df['Method'],drop_first=True)
df = pd.merge(df, method, left_index=True, right_index=True)
df.drop('Method', axis=1, inplace=True)

house_type = pd.get_dummies(df['Type'], drop_first=True)
df = pd.merge(df,house_type, left_index=True, right_index=True)
df.drop('Type', axis=1, inplace=True)

seller = pd.get_dummies(df['SellerG'], drop_first=True)
df = pd.merge(df,seller, left_index=True, right_index=True)
df.drop('SellerG', axis=1, inplace=True)
#print (df)

df.drop('Postcode', axis=1, inplace=True)
df.drop('Address', axis=1, inplace=True)
df.drop('Bedroom2', axis=1, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
    
df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month)
df['Day'] = df['Date'].apply(lambda x: x.day)
    
df['Year'] = df['Year'].replace([2016],'0')
df['Year'] = df['Year'].replace([2017],'12')
df['Year'] = df['Year'].replace([2018],'24')

df = df.apply(pd.to_numeric)
#df["Year"] = pd.to_numeric(df["Year"])
#df["Month"] = pd.to_numeric(df["Month"])

df['Months'] = df['Year'] + df['Month']

df.drop('Year', axis=1, inplace=True)
df.drop('Month', axis=1, inplace=True)
df.drop('Day', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)


df = df.dropna()

print(df.shape)
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#print(df)


X=df.drop('Price', axis=1)
y=df['Price']
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.25, random_state=42)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def Predictive_Model(estimator):
    estimator.fit(train_X, train_y)
    prediction = estimator.predict(train_X)
    print('R_squared_train:', metrics.r2_score(train_y, prediction))
    print('MSE_train:',metrics.mean_squared_error(train_y, prediction))
    prediction = estimator.predict(test_X)
    print('R_squared:', metrics.r2_score(test_y, prediction))
    print('MSE:',metrics.mean_squared_error(test_y, prediction))
    
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print("linear regression")
Predictive_Model(lr)

from sklearn.linear_model import Ridge
rr = Ridge(alpha=100)
print("ridge regression")
Predictive_Model(rr)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
print("5-nn")
Predictive_Model(knn)

knn = KNeighborsRegressor(n_neighbors=10)
print("10-nn")
Predictive_Model(knn)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=15, random_state=0)
print("decision tree")
Predictive_Model(dt)

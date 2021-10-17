import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


dp = pd.read_excel('Pizza_Case.xlsx', sheet_name='Pizza_Case')

#dp=dp.replace({'Customer Satisfaction': {0:1, 1:2, 2:3, 3:4, 4:5, 5:6}})
dp=dp.replace({'Calzone':1, 'Funghi':2, 'Magherita':3,'Paprika':4, 'Salami':5, 'Speciale':6, 'Veggie':7})
dp=dp.replace({'Small': 1, 'Medium': 2, 'Large': 3})
dp=dp.replace({'Chef 1':1, 'Chef 2':2, 'Delivery Guy 1 ':3, 'Delivery Guy 2':4, 'Delivery Scooters ':5, 'Distribution channel fees':6, 'Ingredients':7, 'Phone Bill':8, 'Waiter':9},regex=True)
dp=dp.replace({'Sunday':1, 'Tuesday':2, 'Friday':3, 'Saturday':4, 'Wednesday':5, 'Monday':6, 'Thursday':7,})
dp=dp.replace({'BestOrder Inc.':1, 'Deliver Now Holding':2, 'Deliveruu Inc.':3, 'Feedera SE':4, 'Heropizza Lmtd.':5, 'Orderly SE':6, 'TownExpress Inc. ':7})
dp=dp.replace({'Munich District Five':1, 'Munich District Four':2, 'Munich District One':3, 'Munich District Three':4, 'Munich District Two':5})
dp=dp.replace({'Adult':1, 'Senior':2, 'Student':3, 'Teenager':4})

r=dp['Revenue']
c=dp['Costs']

counter=0
for i in range(len(dp)):
    counter=r[i]-c[i]
    if (counter<0):
        dp.loc[i,'Revenue']=1
    if (counter==0):
        dp.loc[i,'Revenue']=2
    else:
        dp.loc[i,'Revenue']=3

dp=dp.iloc[:,3:len(dp.columns)-1]

labels=np.array(dp['Revenue'])

features= dp.drop(['Revenue'], axis = 1)

feature_list = list(features.columns)

features = np.array(features)


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 5000, random_state = 42)


rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)


errors = abs(predictions - test_labels)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)

accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

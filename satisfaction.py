import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


dp = pd.read_excel('Pizza_Case.xlsx', sheet_name='Pizza_Case')
dp1 = pd.read_excel('Pizza_Event.xlsx', sheet_name='Pizzeria_Event')

dp1=dp1.replace({'Order by phone':1, 'Start preparing pizza':2, 'Call Customer':3,'Start baking pizza':4, 'Baking pizza ready':5, 'Plan route':6, 'Departure pizza':7, 'Pizza arrives at customer':8, 'Payment customer':9, 'Start order website':10, 'Approve order website':11, 'Receive order website':12,'Order at the counter (shop)':13, 'Payment via credit card':14, 'Pizza received':15})
dp1=dp1.replace({'Automation': {'A':1, 'B':2}})
dp=dp.replace({'Customer Satisfaction': {0:1, 1:2, 2:3, 3:4, 4:5, 5:6}})
dp=dp.replace({'Calzone':1, 'Funghi':2, 'Magherita':3,'Paprika':4, 'Salami':5, 'Speciale':6, 'Veggie':7})
dp=dp.replace({'Small': 1, 'Medium': 2, 'Large': 3})
dp=dp.replace({'Chef 1':1, 'Chef 2':2, 'Delivery Guy 1 ':3, 'Delivery Guy 2':4, 'Delivery Scooters ':5, 'Distribution channel fees':6, 'Ingredients':7, 'Phone Bill':8, 'Waiter':9},regex=True)
dp=dp.replace({'Sunday':1, 'Tuesday':2, 'Friday':3, 'Saturday':4, 'Wednesday':5, 'Monday':6, 'Thursday':7,})
dp=dp.replace({'BestOrder Inc.':1, 'Deliver Now Holding':2, 'Deliveruu Inc.':3, 'Feedera SE':4, 'Heropizza Lmtd.':5, 'Orderly SE':6, 'TownExpress Inc. ':7})
dp=dp.replace({'Munich District Five':1, 'Munich District Four':2, 'Munich District One':3, 'Munich District Three':4, 'Munich District Two':5})
dp=dp.replace({'Adult':1, 'Senior':2, 'Student':3, 'Teenager':4})

acten = dp1["ACTIVITY_EN"]

dp = dp.sort_values(by = '_CASE_KEY')

dp = dp.join(acten)
print(dp['ACTIVITY_EN'])

labels=np.array(dp['Customer Satisfaction'])

features= dp.drop(['Customer Satisfaction'], axis = 1)

feature_list = list(features.columns)

features = np.array(features)


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)


errors = abs(predictions - test_labels)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)

accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

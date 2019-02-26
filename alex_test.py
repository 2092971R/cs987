#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:22:17 2019

@author: cmb18162
"""

import pandas as pd
import numpy as np

# Having copied file and stored it locally as a csv file in the working directory
housing = pd.read_csv("BristolAirbnbListings.csv")


housing.plot(kind="scatter", x="id", y="price", ylim = [0,1000])
ax = housing['price'].plot.hist()

housing[['price']].query('price < 300').plot.hist()
# Get some details about it
# First few lines
housing.head()
#housing.info()

#clear non numeric values
housing['bathrooms'].isnull().values.any()
for value in ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'latitude', 'longitude', 'price', 'minimum_nights']:
    housing = housing[pd.to_numeric(housing[value], errors='coerce').notnull()]


housing['accommodates'] = housing.accommodates.astype(float)
housing["accommodates"].where(housing["accommodates"] < 10, 10.0, inplace=True)
housing['bathrooms'] = housing.bathrooms.astype(float)
housing["bathrooms"].where(housing["bathrooms"] < 5, 5.0, inplace=True)
housing['bedrooms'] = housing.bedrooms.astype(float)
housing["bedrooms"].where(housing["bedrooms"] < 5, 5.0, inplace=True)
housing['beds'] = housing.beds.astype(float)
housing["beds"].where(housing["beds"] < 10, 10.0, inplace=True)
housing['review_scores_rating'] = housing.review_scores_rating.astype(float)
housing['latitude'] = housing.latitude.astype(float)
housing['longitude'] = housing.longitude.astype(float)
housing['price'] = housing.price.astype(float)
housing['minimum_nights'] = housing.minimum_nights.astype(float)
housing["minimum_nights"].where(housing["minimum_nights"] < 20, 20.0, inplace=True)
housing['price'] = housing.minimum_nights.astype(float)
housing["price"].where(housing["price"] < 300, 300.0, inplace=True)
housing.info()


list1 = ['Windmill Hill', 'Clifton', 'Southville', 'Bedminster', 'Easton',
       'Ashley', 'Redland', 'Brislington West', 'Brislington East',
       'Lawrence Hill', 'Central', 'Eastville', 'Hotwells & Harbourside',
       'St George West', 'Stoke Bishop', 'Henbury & Brentry', 'Lockleaze',
       'Cotham', 'Southmead', 'Westbury-on-Trym & Henleaze',
       'Clifton Down', 'Stockwood', 'Bishopston & Ashley Down',
       'Frome Vale', 'St George Central', 'Knowle', 'Horfield',
       'Avonmouth & Lawrence Weston', 'Bishopsworth', 'Hillfields',
       'St George Troopers Hill', 'Filwood', 'Hengrove & Whitchurch Park',
       'Hartcliffe & Withywood']
list2 = ['Townhouse', 'Apartment', 'House', 'Guesthouse',
       'Bed and breakfast', 'Barn', 'Loft', 'Hostel', 'Condominium',
       'Guest suite', 'Cabin', 'Tiny house',
       'Serviced apartment', 'Yurt', 'Hut', 'Bungalow', 'Tent',
       'Boat', 'Hotel', 'Cottage', 'Camper/RV',
       "Shepherd's hut (U.K., France)", 'Boutique hotel',
       'Villa', 'Clifton', 'Farm stay',
       'Casa particular (Cuba)']
list3 = ['Private room', 'Entire home/apt', 'Shared room']

housing = housing[housing.neighbourhood.isin(list1)]
housing = housing[housing.property_type.isin(list2)]
housing = housing[housing.room_type.isin(list3)]

# Isolate categorical attribute
from sklearn.preprocessing import OneHotEncoder
for value in ['room_type']:
    housing_temp = housing[value]
    housing_encoded, housing_categories = housing_temp.factorize()
    encoder = OneHotEncoder()
    housing_1hot = encoder.fit_transform(housing_encoded.reshape(-1,1))
    hcea = housing_1hot.toarray()
    housing[value] = hcea
    enc_data = pd.DataFrame(housing_1hot.toarray())
    enc_data.columns = housing_categories
    enc_data.index = housing.index
    housing_num = housing.drop(value, axis=1)
    housing = housing_num.join(enc_data)

housing_labels = housing["price"].copy()
housing = housing.drop(["price", "minimum_nights", "id", "name", "host_id", "host_name", "postcode", "number_of_reviews", "last_review", "reviews_per_month","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication", "review_scores_location","review_scores_value" , "calculated_host_listings_count","availability_365", 'neighbourhood', 'property_type'], axis=1) 

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing, housing_labels)

scores = cross_val_score(lin_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=7)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
housing.info()

housing.plot(kind="scatter", x="price", y="accommodates")
plt.show()

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
scores = cross_val_score(sgd_clf, housing, housing_labels, scoring="neg_mean_squared_error", cv=3)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(housing, housing_labels)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(housing, housing_labels, random_state=42)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='hard')

voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42) 

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='soft')

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


"""

housing.info()
housing["bathrooms"].hist()
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
splits = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
splits.get_n_splits(housing, housing_labels)
print(splits)
for train_index , test_index in splits.split(housing, housing_labels):
    X_train, X_test = housing[train_index], housing[test_index]
    Y_train, Y_test = housing_labels[train_index], housing_labels[test_index]


>>> splits = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
>>> sss.get_n_splits(X, y)
5
>>> print(sss)       
StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
>>> for train_index, test_index in sss.split(X, y):
...    print("TRAIN:", train_index, "TEST:", test_index)
...    X_train, X_test = X[train_index], X[test_index]
...    y_train, y_test = y[train_index], y[test_index]
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


"""









# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:52:19 2024

@author: SAMI
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

data = pd.read_csv('final_loan_sansaction.csv')
data  = data.drop('Unnamed: 0' , axis=1)
X = data.drop('Loan_Status' , axis=1)
y = data['Loan_Status']


# Assuming X and y are your features and target variable
smote = SMOTE()
X, y = smote.fit_resample(X, y)

model = RandomForestClassifier(bootstrap=True , n_estimators=64 , max_features=4)
model.fit(X,y)

joblib.dump(model, 'loan_santion')
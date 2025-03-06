import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Charger le dataset en un Dataframe Pandas
credit_card_data=pd.read_csv('creditcard.csv')


#Les 5 premières lignes du dtaset
print(credit_card_data.head())
print(credit_card_data.tail())

#Dataset infos
print(credit_card_data.info())

#Cheking the nuumber of missing values in each column
credit_card_data.isnull().sum()
#distribution of legit and frodulous ones
print(credit_card_data['Class'].value_counts())

#Separation of the dat for analysis
legit=credit_card_data[credit_card_data.Class ==0]
fraud=credit_card_data[credit_card_data.Class ==1]
print(legit.shape)
print(fraud.shape)

#Statistical mesures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
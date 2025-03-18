import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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


#Comparing the values 
print(credit_card_data.groupby('Class').mean())

#Creation of a balanced dataset
legit_sample=legit.sample(n=492)


#Concatenation of two dataset
new_dataset=pd.concat([legit_sample,fraud],axis=0)

print(new_dataset.head())

print(new_dataset['Class'].value_counts())

#To see if our sample is correct (correct is the values are almost the same tha those of the dataset)
print(new_dataset.groupby('Class').mean())


#Découpage du dataset en features et targets
#Axis 0=Row et Axis 1=Column
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset["Class"]

print(X)
print(Y)



#découpage en données d'entraînement et en données de Test

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)
print(Y_train.value_counts())




#Entraînement du modèle avec les données d'entrainement



#Normalisation des données pour améliorer la convergence

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)


#Test de la performance du model

X_train_prediction=model.predict(X_train_scaled)
precision_model_training=accuracy_score(X_train_prediction,Y_train)

print("Exactitude sur des données: " ,precision_model_training)


#Evaluation de l'exactitude sur des données non utilisées
X_test_prediction=model.predict(X_test_scaled)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


print("Exactitude sur des données de test: " ,test_data_accuracy)

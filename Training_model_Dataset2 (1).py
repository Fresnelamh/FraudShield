import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler





#Manipulation du modèle  en utilisant le datset ieee-fraud-detection


#Charger le dataset en un Dataframe Pandas
"""
train_transaction_data2=pd.read_csv(os.path.join("..","Data","Dataset2","train_transaction.csv"))
train_identity_data2=pd.read_csv(os.path.join("..","..","Data","Dataset2","train_identity.csv"))
#pd.read_csv('../../Data/Dataset2/train_identity.csv')
test_transaction_data2=pd.read_csv(os.path.join("..","..","Data","Dataset2","test_transaction.csv"))
#pd.read_csv('../../Data/Dataset2/test_transaction.csv')
test_identity_data2=pd.read_csv(os.path.join("..","..","Data","Dataset2","test_identity.csv"))
#pd.read_csv('../../Data/Dataset2/test_identity.csv')"
"""""
""
train_transaction_data2=pd.read_csv('train_transaction.csv')
train_identity_data2=pd.read_csv('train_identity.csv')
test_transaction_data2=pd.read_csv('test_transaction.csv')
test_identity_data2=pd.read_csv('test_identity.csv')



#Fusionner les données des transactions et les données identités

print(f"Données de transaction:\n{train_transaction_data2.head()}")
print(f"Données idendité:\n{train_identity_data2.head()}")
print(f"Données de transaction:\n{train_transaction_data2.tail()}")
print(f"Données idendité:\n{train_identity_data2.tail()}")


# Fusionner les datasets en effectuant une jointure gauche (left join)
train_data_concat = train_transaction_data2.merge(train_identity_data2, on="TransactionID", how="left")


print(f"Données fusionnées:\n{train_data_concat.head()}")  # Affiche les premières lignes du dataset fusionné
print(train_data_concat.info())  # Vérifie les colonnes et les valeurs manquantes






#Nombre de transactions par classe

print(train_transaction_data2['isFraud'].value_counts())



#Gestion des valeurs manquantes du dataset



#Mesurer le pourcentage de données manquantes chaque colonne du dataset contient 
missing_values = train_data_concat.isnull().sum()
missing_percentage = (missing_values / len(train_data_concat)) * 100
print(missing_percentage.sort_values(ascending=False))




#Supprimer les colonnes avec plus de 70% de valeurs manquantes
threshold = 70 

# Trouver les colonnes à supprimer
cols_to_drop = missing_percentage[missing_percentage > threshold].index

# Supprimer ces colonnes du dataset
train_data_concat.drop(columns=cols_to_drop, inplace=True)

 # Vérifier combien de colonnes ont été supprimées
print(f"Colonnes supprimées : {len(cols_to_drop)}") 

# Vérifier combien il en reste
print(f"Nouveau nombre de colonnes : {train_data_concat.shape[1]}")  




# Remplir les valeurs manquantes numériques avec la médiane
for col in train_data_concat.select_dtypes(include=['number']).columns:
   # train_data_concat[col].fillna(train_data_concat[col].median(), inplace=True)
    train_data_concat[col] = train_data_concat[col].fillna(train_data_concat[col].median())


# Remplir les valeurs manquantes catégorielles avec la valeur la plus fréquente (mode)
for col in train_data_concat.select_dtypes(include=['object']).columns:
   # train_data_concat[col].fillna(train_data_concat[col].mode()[0], inplace=True)
    train_data_concat[col] = train_data_concat[col].fillna(train_data_concat[col].mode()[0])



print(f" Nombre de données manquantes à la fin :{train_data_concat.isnull().sum().sum()}")


print(train_data_concat.head())
print(train_data_concat.tail())

zero_counts = (train_data_concat == 0).sum()
zero_percent = (zero_counts / len(train_data_concat)) * 100
print(zero_percent[zero_percent > 50])  # Affiche les colonnes où plus de 50% des valeurs sont 0


#Rééquilibrage du dataset 

#Distribution des transaction légitimes et celles frauduleuses
print(train_data_concat['isFraud'].value_counts())

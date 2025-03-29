import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,confusion_matrix,precision_score, recall_score






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


#Encodage des valeurs catégorielles





#Rééquilibrage du dataset 

#Distribution des transaction légitimes et celles frauduleuses
print(train_data_concat['isFraud'].value_counts())



legit=train_data_concat[train_data_concat.isFraud ==0]
fraud=train_data_concat[train_data_concat.isFraud ==1]
print(legit.shape)
print(fraud.shape)

#Statistical mesures of the data
print(legit.TransactionAmt.describe())
print(fraud.TransactionAmt.describe())


print(train_data_concat['isFraud'].dtype)


#Comparing the values 
#print(train_data_concat.groupby('isFraud').mean())

#Creation of a balanced dataset
legit_sample=legit.sample(n=20663)


print(legit_sample.head())
print(fraud.head())




#Concatenation of two dataset
new_dataset2=pd.concat([legit_sample,fraud],axis=0)


#Mélange aléatoire des transactions frauduleuses et celles légitimes

new_dataset2 = new_dataset2.sample(frac=1, random_state=42).reset_index(drop=True)



print(new_dataset2['isFraud'].value_counts())



print(new_dataset2.head())




# Identifier les colonnes contenant du texte
colonnes_textuelles = list(new_dataset2.select_dtypes(include=['object', 'string']).columns)

print("Colonnes textuelles:" ,colonnes_textuelles)







X=new_dataset2.drop(columns='isFraud',axis=1)
Y=new_dataset2["isFraud"]
print("X:")
print(X)
print("Y:")
print(Y)



#Division des données en données de test et d'entrainement 

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)
print(Y_train.value_counts())




#Normalisation des données pour améliorer la convergence




#Test de la performance du model







# Initialiser le modèle
catboost_model = CatBoostClassifier(iterations=500,  # Nombre d'itérations
                                    learning_rate=0.1,  # Taux d'apprentissage
                                    depth=5,  # Profondeur des arbres
                                    cat_features=colonnes_textuelles,  # Colonnes catégorielles
                                    verbose=100)  # Affichage des logs chaque 100 itérations

# Entraîner le modèle
catboost_model.fit(X_train, Y_train, cat_features=colonnes_textuelles)

# Prédictions sur les données d'entraînement
y_train_pred = catboost_model.predict(X_train)

# Évaluer la performance sur l'entraînement
#print("Accuracy sur les données d'entraînement:", accuracy_score(Y_train, y_train_pred))



#Prédictions sur les données du test
y_test_pred = catboost_model.predict(X_test)


# Évaluer la performance sur l'entraînement
#print("Accuracy sur les données d'entraînement:", accuracy_score(Y_test, y_test_pred))


#Evaluer le modèle avec F1-score
precision_train = precision_score(Y_test, y_train_pred)
precision_test = precision_score(Y_test, y_test_pred)
recall_train = recall_score(Y_test, y_train_pred)
recall_test = recall_score(Y_test, y_test_pred)
f1_train = f1_score(Y_train, y_train_pred)
f1_test = f1_score(Y_test, y_test_pred)  


print(f"Precision sur le train set: {precision_train:.4f}")
print(f"Precision  sur le test set: {precision_test:.4f}")
print(f"Recall sur le train set: {recall_train:.4f}")
print(f"Recall sur le test set: {recall_test:.4f}")


print(f"F1-Score sur le train set: {f1_train:.4f}")
print(f"F1-Score sur le test set: {f1_test:.4f}")


#Afficher les faux positifs et faux négatifs
cm = confusion_matrix(Y_test, y_test_pred)
print("Matrice de confusion :\n", cm)





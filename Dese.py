import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler







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
"""
'''
train_transaction_data2=pd.read_csv('train_transaction.csv',nrows=5000)
train_identity_data2=pd.read_csv('train_identity.csv',nrows=5000)
test_transaction_data2=pd.read_csv('test_transaction.csv',nrows=5000)
test_identity_data2=pd.read_csv('test_identity.csv',nrows=5000)
'''

train_transaction_data2=pd.read_csv('train_transaction.csv')
train_identity_data2=pd.read_csv('train_identity.csv')
test_transaction_data2=pd.read_csv('test_transaction.csv')



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









#Mélange aléatoire des transactions frauduleuses et celles légitimes


train_data_concat=train_data_concat.drop(columns=['TransactionID','TransactionDT'])



print(train_data_concat['isFraud'].value_counts())



print(train_data_concat.head())




# Identifier les colonnes contenant du texte

colonnes_textuelles = list(train_data_concat.select_dtypes(include=['object', 'string']).columns)


print("Colonnes textuelles:" ,colonnes_textuelles)



for col in train_data_concat.columns:
    # Essayer de convertir la colonne en numérique
    train_data_concat[col] = pd.to_numeric(train_data_concat[col], errors='coerce')

# Vérifie les lignes où les valeurs sont NaN (indiquant une conversion échouée)
rows_with_na = train_data_concat[train_data_concat.isna().any(axis=1)]
print(rows_with_na)
#Mélange aléatoire des lignes
train_data_concat=train_data_concat.sample(frac=1, random_state=42).reset_index(drop=True)




X=train_data_concat.drop(columns='isFraud',axis=1)
Y=train_data_concat["isFraud"]
print("X:")
print(X)
print("Y:")
print(Y)








#Division des données en données de test et d'entrainement 

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)
print(Y_train.value_counts())


print(X_train.dtypes)
le = LabelEncoder()

for col in colonnes_textuelles:
    X_train[col] = le.fit_transform(X_train[col])

    X_test[col] = le.fit_transform(X_test[col])



#Over-sampling
'''
smote = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled =smote.fit_resample(X_train, Y_train)
model = CatBoostClassifier(verbose=0)'''


rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, Y_train)
'''
smote = SMOTETomek(random_state=42)
X_resampled, y_resampled =smote.fit_resample(X_train, Y_train)


'''

'''
# Définition des modèles
ratio = Y_train.value_counts()[0] / Y_train.value_counts()[1]
log_clf = LogisticRegression(class_weight='balanced',max_iter=1500)
xgb_clf = XGBClassifier(scale_pos_weight=ratio,use_label_encoder=False, eval_metric='logloss')
cat_clf = CatBoostClassifier(auto_class_weights='Balanced',verbose=0)

# Voting Classifier (vote par probabilité)
voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf),
    ('xgb', xgb_clf),
    ('cat', cat_clf)
], voting='soft',
n_jobs=1)  # 'soft' = moyenne des probabilités

# Entraînement
voting_clf.fit(X_under, y_under)






# Prédictions
y_pred = voting_clf.predict(X_test)'''
'''
ratio = Y_train.value_counts()[0] / Y_train.value_counts()[1]

model = XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, Y_train)

model.fit(X_resampled, y_resampled)


seuils=[0.25,0.3,0.40,0.60,0.70]

for seuil in seuils:
 y_pred = (y_proba > seuil).astype(int)  # ou 0.25 si tu veux plus de sensibilité
 print("Classification Report - Class Weights:")
 print(classification_report(Y_test, y_pred))
 y_pred = model.predict(X_test)
 '''



# Define CatBoost model with class weights
model_class_weights = CatBoostClassifier(auto_class_weights="Balanced", random_state=42, verbose=0,cat_features=colonnes_textuelles)
model_class_weights.fit(X_under, y_under)
y_pred_class_weights = model_class_weights.predict(X_test)
print("Classification Report - Class Weights:")
print(classification_report(Y_test, y_pred_class_weights))

print("Classification Report - Class Weights:")
print(classification_report(Y_test, y_pred_class_weights))
f1 = f1_score(Y_test, y_pred_class_weights)
print(f"\nF1 Score: {f1:.4f}")

# Calcul de la précision pour la classe frauduleuse (1)
precision = precision_score(Y_test, y_pred_class_weights)
print(f"\nPrecision: {precision:.4f}")

# Calcul du rappel (recall) pour la classe frauduleuse (1)
recall = recall_score(Y_test, y_pred_class_weights)
print(f"\nRecall: {recall:.4f}")
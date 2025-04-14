import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
credit_card_data = pd.read_csv('creditcard.csv')
print(credit_card_data['Class'].value_counts())

# Mélange des données
credit_card_data = credit_card_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Séparation features/target
X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']

# Découpage en données d'entraînement et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Oversampling avec SMOTETomek
smk = SMOTETomek(random_state=42)
X_resampled, Y_resampled = smk.fit_resample(X_train, Y_train)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Calcul du scale_pos_weight
neg = np.sum(Y_resampled == 0)
pos = np.sum(Y_resampled == 1)
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# Modèle XGBoost
model = XGBClassifier(scale_pos_weight=scale_pos_weight,
                      use_label_encoder=False,
                      eval_metric='logloss',
                      random_state=42)

model.fit(X_train_scaled, Y_resampled)



'''
seuils=[0.25,0.3,0.40,0.60,0.70]

for seuil in seuils:
 y_pred = (y_proba > seuil).astype(int)  # ou 0.25 si tu veux plus de sensibilité
 print("Classification Report - Class Weights:")
 print(classification_report(Y_test, y_pred))
 y_pred = model.predict(X_test)

 '''
# Prédictions
Y_pred = model.predict(X_test_scaled)

# Évaluations
acc = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test_scaled)[:, 1])

print("=== ÉVALUATION ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC ROC : {roc_auc:.4f}")

print("=== Classification Report ===")
print(classification_report(Y_test, Y_pred))

# Matrice de confusion
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.show()

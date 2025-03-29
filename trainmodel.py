import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ Étape 1 : Génération de données factices
np.random.seed(42)  # Pour des résultats reproductibles
X = np.random.rand(1000, 5)  # 1000 transactions avec 5 caractéristiques chacune
y = np.random.randint(0, 2, 1000)  # 0 = normale, 1 = frauduleuse

# ✅ Étape 2 : Division en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Étape 3 : Création et entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Étape 4 : Sauvegarde du modèle dans un fichier "fraud_model.pkl"
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle de détection de fraude sauvegardé dans 'fraud_model.pkl'")

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = joblib.load('fraud_detection_model.pkl')  # Remplacez par le chemin vers votre modèle

@app.route('/transaction', methods=['POST'])
def analyze_transaction():
    # Récupérer les données de la requête
    
    data = request.get_json()

    # Extraire les informations pertinentes de la transaction
    amount = data.get('amount')
    recipient = data.get('recipient')
    location = data.get('location')
    user_id = data.get('user_id')
    # Appliquer des transformations ou des prétraitements (si nécessaire)
    features = np.array([amount, location, user_id])  # Exemple simple, adapter selon vos données

    # Prédire avec le modèle
    prediction = model.predict([features])

    if prediction == 1:  # Si la transaction est frauduleuse
        return jsonify({'status': 'fraud_detected', 'message': 'Transaction suspecte détectée. Veuillez confirmer ou annuler.'}), 400
    else:
        return jsonify({'status': 'approved', 'message': 'Transaction validée.'}), 200

if __name__ == '__main__':
    app.run(debug=True)


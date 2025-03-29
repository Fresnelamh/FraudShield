from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from database import get_db_connection


app = FastAPI()



# Charger le modèle ML
with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    

# Modèle de transaction
class Transaction(BaseModel):
    id: int
    montant: float
    devise: str
    pays: str
    utilisateur_id: int

@app.post("/detect_fraude/")
def detecter(transaction: Transaction):
    conn = get_db_connection()
    cur = conn.cursor()

    # Vérifier l'historique des transactions de l'utilisateur
    cur.execute("SELECT COUNT(*) AS count FROM transactions WHERE utilisateur_id = %s", (transaction.utilisateur_id,))
    historique = cur.fetchone()["count"]

    # Préparer les features pour le modèle ML
    features = np.array([[transaction.montant, transaction.utilisateur_id, len(transaction.devise), ord(transaction.pays[0]), historique]])
    prediction = model.predict(features)[0]

    # Si fraude détectée, insérer dans la base de données
    if prediction == 1:
        cur.execute(
            "INSERT INTO transactions_frauduleuses (id, montant, pays, utilisateur_id) VALUES (%s, %s, %s, %s)",
            (transaction.id, transaction.montant, transaction.pays, transaction.utilisateur_id)
        )
        conn.commit()

    conn.close()

    if prediction == 1:
        raise HTTPException(status_code=400, detail="Transaction suspecte !")

    return {"message": "Transaction valide", "transaction_id": transaction.id}

@app.get("/transactions_suspectes/")
def get_transactions_suspectes():
    conn = get_db_connection()
    cur = conn.cursor()

    # Récupérer les transactions suspectes
    cur.execute("SELECT * FROM transactions_frauduleuses")
    transactions = cur.fetchall()

    conn.close()

    if not transactions:
        raise HTTPException(status_code=404, detail="Aucune transaction suspecte trouvée")

    return {"transactions": transactions}



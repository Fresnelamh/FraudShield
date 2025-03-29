import numpy as np
from datetime import datetime, timedelta

def prepare_features(transaction, conn):
    cur = conn.cursor()

    
    cur.execute("""
        SELECT COUNT(*) 
        FROM transactions 
        WHERE utilisateur_id = %s AND date > %s
    """, (transaction.utilisateur_id, datetime.now() - timedelta(days=30)))
    recent_transactions_count = cur.fetchone()[0]

    
    cur.execute("""
        SELECT AVG(montant) 
        FROM transactions 
        WHERE utilisateur_id = %s
    """, (transaction.utilisateur_id,))
    avg_transaction_amount = cur.fetchone()[0]

    
    devise_codes = {'USD': 0, 'EUR': 1, 'GBP': 2, 'INR': 3}  
    devise_code = devise_codes.get(transaction.devise, -1) 

   
    cur.execute("""
        SELECT COUNT(DISTINCT pays) 
        FROM transactions 
        WHERE utilisateur_id = %s
    """, (transaction.utilisateur_id,))
    countries_count = cur.fetchone()[0]

    
    features = np.array([
        [transaction.montant, 
         transaction.utilisateur_id, 
         recent_transactions_count, 
         avg_transaction_amount, 
         devise_code, 
         countries_count]
    ])

    return features

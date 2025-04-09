import sqlite3

conn = sqlite3.connect("fraude.db")
cursor = conn.cursor()

# Crée la table principale
cursor.execute('''
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    montant REAL,
    devise TEXT,
    pays TEXT,
    utilisateur_id INTEGER
)
''')

# Crée la table des fraudes
cursor.execute('''
CREATE TABLE IF NOT EXISTS transactions_frauduleuses (
    id INTEGER PRIMARY KEY,
    montant REAL,
    pays TEXT,
    utilisateur_id INTEGER
)
''')

conn.commit()
conn.close()

print("✅ Base de données SQLite initialisée")

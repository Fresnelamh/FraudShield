import pymysql

# Connexion MySQL
def get_db_connection():
    conn = pymysql.connect(
        host="localhost",
        user="admin",
        password="secret",
        database="fraude",
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def envoyer_alerte_email(destinataire, transaction):
    expediteur = "leilaeffiboley411@gmail.com"  # Remplace avec ton email
    mot_de_passe = "tlxq hzlj ukqi lmtd"        # Mot de passe d'application Gmail

    message = MIMEMultipart("alternative")
    message["Subject"] = "🚨 Alerte : Transaction suspecte détectée"
    message["From"] = expediteur
    message["To"] = destinataire

    texte = f"""
    Alerte ! Une transaction suspecte vient d'être détectée :
    
    ID : {transaction.id}
    Montant : {transaction.montant} {transaction.devise}
    Pays : {transaction.pays}
    Utilisateur ID : {transaction.utilisateur_id}
    """

    message.attach(MIMEText(texte, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as serveur:
            serveur.login(expediteur, mot_de_passe)
            serveur.sendmail(expediteur, destinataire, message.as_string())
        print("✅ Alerte email envoyée à", destinataire)
    except Exception as e:
        print("❌ Échec de l'envoi de l'email :", e)

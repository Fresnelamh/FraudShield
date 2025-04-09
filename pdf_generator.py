from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generer_pdf(transaction_id, montant, date_transaction, nom_client, statut, commentaire, dossier="rapports"):
    if not os.path.exists(dossier):
        os.makedirs(dossier)

    nom_fichier = f"{dossier}/rapport_transaction_{transaction_id}.pdf"
    
    c = canvas.Canvas(nom_fichier, pagesize=A4)
    largeur, hauteur = A4

    # Titre principal
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, hauteur - 50, "🚨 Rapport de Détection de Fraude")

    # Détails
    c.setFont("Helvetica", 12)
    c.drawString(50, hauteur - 100, f"🆔 ID de la transaction : {transaction_id}")
    c.drawString(50, hauteur - 120, f"👤 Client : {nom_client}")
    c.drawString(50, hauteur - 140, f"💰 Montant : {montant} €")
    c.drawString(50, hauteur - 160, f"📅 Date : {date_transaction}")
    c.drawString(50, hauteur - 180, f"🔒 Statut : {'⚠️ FRAUDE détectée' if statut else '✅ Transaction normale'}")

    # Commentaire
    c.drawString(50, hauteur - 220, "📝 Commentaire :")
    texte = commentaire or "Aucun commentaire."
    c.drawString(70, hauteur - 240, texte)

    # Date de génération du rapport
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

    c.save()
    return nom_fichier

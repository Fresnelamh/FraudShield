
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="FraudShield by Ambiton", layout="wide")

st.title("🚨 FraudShield by Ambiton")
st.markdown("Un tableau de bord interactif pour la détection de fraudes bancaires.")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Transactions", "Alertes"])

# Dummy data for testing
def load_data():
    return pd.DataFrame([
        {"id": 1, "montant": 1200.0, "pays": "FR", "utilisateur_id": 42, "fraude": True},
        {"id": 2, "montant": 80.0, "pays": "US", "utilisateur_id": 13, "fraude": False},
        {"id": 3, "montant": 3000.0, "pays": "NG", "utilisateur_id": 42, "fraude": True},
        {"id": 4, "montant": 450.0, "pays": "DE", "utilisateur_id": 55, "fraude": False},
    ])

df = load_data()

if page == "Dashboard":
    st.subheader("📊 Vue d'ensemble")
    col1, col2 = st.columns(2)

    with col1:
        total_fraudes = df[df["fraude"] == True].shape[0]
        st.metric("🚨 Fraudes détectées", total_fraudes)

    with col2:
        total_montant = df[df["fraude"] == True]["montant"].sum()
        st.metric("💸 Montant frauduleux", f"{total_montant} €")

    fig = px.histogram(df, x="pays", color="fraude", title="Répartition des transactions par pays")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Transactions":
    st.subheader("📄 Liste des transactions")
    filtre = st.selectbox("Filtrer par", ["Toutes", "Frauduleuses", "Non frauduleuses"])

    if filtre == "Frauduleuses":
        data = df[df["fraude"] == True]
    elif filtre == "Non frauduleuses":
        data = df[df["fraude"] == False]
    else:
        data = df

    st.dataframe(data)

elif page == "Alertes":
    st.subheader("🚨 Alerte transaction suspecte")
    alert_id = st.number_input("ID de la transaction à examiner", min_value=1, step=1)
    tx = df[df["id"] == alert_id]

    if not tx.empty:
        st.write(tx)
        st.markdown("## Action utilisateur")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ C'était moi"):
                st.success("Confirmation reçue.")
        with col2:
            if st.button("❌ Ce n'était pas moi"):
                st.error("Fraude signalée.")
    else:
        st.warning("Transaction non trouvée.")

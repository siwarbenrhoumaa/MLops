import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ============================================================================
#                           CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Crime Prediction LA 2020", page_icon="🚨", layout="wide")

API_URL = "http://localhost:8000"

DAYS_MAP = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
MONTHS_MAP = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
              7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}

CRIME_COLORS = {
    "Violent Crime": "#d62728",
    "Property & Theft Crime": "#ff7f0e",
    "Vehicle-Related Crime": "#2ca02c",
    "Other / Fraud / Public Order Crime": "#9467bd"
}

# ============================================================================
#                           UTILITAIRES
# ============================================================================

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200 and r.json()["model_loaded"]
    except:
        return False

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        return None

def predict(features):
    try:
        r = requests.post(f"{API_URL}/predict", json=features, timeout=15)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API {r.status_code}")
        return None
    except Exception as e:
        st.error(f"Connexion impossible : {e}")
        return None

# ============================================================================
#                           SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🚨 Crime Prediction LA")
    if check_api():
        st.success("✅ Connecté")
    else:
        st.error("❌ API indisponible")

    model_info = get_model_info()
    if model_info:
        st.metric("Version", model_info['version'])
        st.metric("Accuracy", f"{model_info['metrics'].get('test_accuracy', 0):.1%}")

    page = st.radio("Menu", ["Accueil", "Prédiction", "Batch"])

# ============================================================================
#                           PAGES
# ============================================================================

if page == "Accueil":
    st.title("🚨 Prédiction des Crimes - Los Angeles 2020")
    st.markdown("Système MLOps complet pour prédire le type de crime")

    if model_info:
        st.success(f"Modèle v{model_info['version']} chargé - Accuracy {model_info['metrics'].get('test_accuracy', 0):.1%}")

    st.markdown("**Utilisez 'Prédiction' pour tester en direct !**")

elif page == "Prédiction":
    st.title("🎯 Prédiction Individuelle")

    with st.form("form"):
        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Heure", 0, 23, 12)
            day = st.selectbox("Jour", options=list(DAYS_MAP.keys()), format_func=lambda x: DAYS_MAP[x])
            month = st.selectbox("Mois", options=list(MONTHS_MAP.keys()), format_func=lambda x: MONTHS_MAP[x])
            lat = st.number_input("Latitude", 33.0, 35.0, 34.0522, format="%.4f")
            lon = st.number_input("Longitude", -119.0, -117.0, -118.2437, format="%.4f")

        with col2:
            age = st.number_input("Âge victime", 0, 120, 35)
            area = st.slider("Zone LAPD", 1, 21, 12)
            sex = st.selectbox("Sexe", ["", "M", "F", "X"])
            descent = st.selectbox("Descent", ["", "H", "B", "W", "A", "O"])
            premis = st.number_input("Code lieu", value=101.0)
            part = st.selectbox("Part 1-2", [1, 2])

        submitted = st.form_submit_button("🔮 Prédire", type="primary")

    if submitted:
        payload = {
            "Hour": hour,
            "Day_of_week": day,
            "Month_num": month,
            "LAT": lat,
            "LON": lon,
            "Vict_Age": age,
            "AREA": area,
            "Vict_Sex": sex if sex else None,
            "Vict_Descent": descent if descent else None,
            "Premis_Cd": premis,
            "Part_1_2": part
        }

        with st.spinner("Prédiction..."):
            result = predict(payload)

        if result:
            crime = result['predicted_crime_group']
            color = CRIME_COLORS.get(crime, "#333")
            st.markdown(f"<h1 style='text-align:center; color:{color}'>{crime.upper()}</h1>", unsafe_allow_html=True)
            st.metric("Confiance", f"{result['confidence']:.1%}")
            st.info(f"Version modèle : {result['model_version']}")

elif page == "Batch":
    st.title("📊 Batch")
    st.info("CSV avec colonnes : Hour, Day_of_week, Month_num, LAT, LON, Vict_Age, AREA, Vict_Sex, Vict_Descent, Premis_Cd, Part_1_2")

    uploaded = st.file_uploader("CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Lancer Batch"):
            with st.spinner("Traitement..."):
                result = requests.post(f"{API_URL}/predict/batch", json=df.to_dict(orient="records")).json()
            st.success(f"{result['total_predictions']} prédictions")
            # Visualisation possible ici

st.markdown("---")
st.caption("Projet MLOps 2025 - MLflow + FastAPI + Streamlit")
"""
Interface Streamlit pour la prédiction de crimes à Los Angeles
Version Pipeline Automatisé - Auto-refresh des infos du modèle
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ============================================================================
#                           CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crime Prediction LA",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (configurable)
API_URL = "http://localhost:8000"

# Mappings
DAYS_MAP = {
    0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi",
    4: "Vendredi", 5: "Samedi", 6: "Dimanche"
}

MONTHS_MAP = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

CRIME_COLORS = {
    "Violent Crime": "#d62728",
    "Property & Theft Crime": "#ff7f0e",
    "Vehicle-Related Crime": "#2ca02c",
    "Other / Fraud / Public Order Crime": "#9467bd"
}

CRIME_EMOJIS = {
    "Violent Crime": "⚔️",
    "Property & Theft Crime": "🏠",
    "Vehicle-Related Crime": "🚗",
    "Other / Fraud / Public Order Crime": "📋"
}

# ============================================================================
#                           UTILITAIRES API
# ============================================================================

@st.cache_data(ttl=60)  # Cache pendant 60 secondes
def check_api():
    """Vérifie si l'API est accessible"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


@st.cache_data(ttl=30)  # Refresh toutes les 30 secondes
def get_model_info():
    """Récupère les informations du modèle en production"""
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        return None


@st.cache_data(ttl=60)
def get_health():
    """Récupère l'état de santé de l'API"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=50)
        if r.status_code == 200:
            return r.json()
    except:
        return None


def predict(features):
    """Appelle l'API pour faire une prédiction"""
    try:
        r = requests.post(f"{API_URL}/predict", json=features, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"❌ Erreur API {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"❌ Connexion impossible : {e}")
        return None


def reload_model():
    """Force le rechargement du modèle"""
    try:
        r = requests.post(f"{API_URL}/reload-model")
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None


# ============================================================================
#                           SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🚨 Crime Prediction LA")
    st.markdown("---")
    
    # Statut de l'API
    if check_api():
        st.success("✅ API Connectée")
        
        # Informations du modèle
        health = get_health()
        model_info = get_model_info()
        
        if model_info:
            st.markdown("### 📊 Modèle en Production")
            st.metric("Version", f"v{model_info['version']}")
            
            accuracy = model_info['metrics'].get('test_accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.1%}", delta=f"+{(accuracy-0.5)*100:.1f}% vs baseline")
            
            st.metric("F1-Score", f"{model_info['metrics'].get('test_f1_weighted', 0):.3f}")
            
            # Run ID (tronqué)
            if model_info.get('run_id'):
                st.text(f"Run ID: {model_info['run_id'][:12]}...")
            
            # Dernière mise à jour
            if model_info.get('last_updated'):
                last_updated = datetime.fromisoformat(model_info['last_updated'])
                st.text(f"Mis à jour: {last_updated.strftime('%d/%m/%Y %H:%M')}")
        
        if health:
            st.markdown("### 📈 Statistiques")
            st.metric("Prédictions", health.get('total_predictions', 0))
            
            if health.get('last_check'):
                last_check = datetime.fromisoformat(health['last_check'])
                st.text(f"Vérifié: {last_check.strftime('%H:%M:%S')}")
        
        # Bouton de rechargement manuel
        st.markdown("---")
        if st.button("🔄 Recharger Modèle", use_container_width=True):
            with st.spinner("Rechargement..."):
                result = reload_model()
                if result:
                    st.success(f"✅ {result['message']}")
                    if result.get('new_version'):
                        st.info(f"Nouvelle version: v{result['new_version']}")
                    # Invalider le cache
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Échec du rechargement")
    else:
        st.error("❌ API Indisponible")
        st.info("Vérifiez que l'API est démarrée sur le port 8000")
    
    st.markdown("---")
    
    # Menu de navigation
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🎯 Prédiction", "📊 Batch", "📈 Statistiques"],
        label_visibility="collapsed"
    )

# ============================================================================
#                           PAGE : ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    st.title("🚨 Prédiction des Crimes - Los Angeles")
    st.markdown("### Système MLOps Complet avec Pipeline Automatisé")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 Intelligence Artificielle")
        st.write("Modèles avancés (RF, XGBoost, LightGBM)")
        st.write("Ensembles (Voting, Stacking)")
        st.write("Auto-sélection du meilleur modèle")
    
    with col2:
        st.markdown("#### 🔄 Pipeline Automatisé")
        st.write("Preprocessing automatique")
        st.write("Drift detection")
        st.write("Réentraînement si nécessaire")
    
    with col3:
        st.markdown("#### 🚀 Production")
        st.write("Déploiement automatique")
        st.write("API FastAPI")
        st.write("Interface Streamlit")
    
    st.markdown("---")
    
    # Informations détaillées du modèle
    model_info = get_model_info()
    if model_info:
        st.markdown("### 📊 Modèle Actuel en Production")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Métriques de Performance")
            metrics = model_info['metrics']
            
            # Gauge pour l'accuracy
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get('test_accuracy', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Test Accuracy (%)"},
                delta={'reference': 50, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("F1-Score", f"{metrics.get('test_f1_weighted', 0):.4f}")
            st.metric("CV Mean", f"{metrics.get('cv_accuracy_mean', 0):.4f}")
        
        with col2:
            st.markdown("#### Informations Techniques")
            st.text(f"Nom : {model_info['model_name']}")
            st.text(f"Version : v{model_info['version']}")
            st.text(f"Stage : {model_info['stage']}")
            
            if model_info.get('run_id'):
                st.text(f"Run ID : {model_info['run_id'][:16]}...")
            
            st.markdown("**Features utilisées:**")
            for feat in model_info['features_used'][:5]:
                st.text(f"  • {feat}")
            if len(model_info['features_used']) > 5:
                st.text(f"  ... et {len(model_info['features_used']) - 5} autres")
            
            st.markdown("**Classes prédites:**")
            for i, cls in enumerate(model_info['classes']):
                emoji = CRIME_EMOJIS.get(cls, "📋")
                st.text(f"  {emoji} {cls}")
    
    st.markdown("---")
    st.info("💡 **Utilisez 'Prédiction' dans le menu pour tester le modèle en temps réel !**")


# ============================================================================
#                           PAGE : PRÉDICTION
# ============================================================================

elif page == "🎯 Prédiction":
    st.title("🎯 Prédiction Individuelle")
    st.markdown("Entrez les caractéristiques d'un crime pour prédire son type")
    
    with st.form("form_prediction"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Informations Temporelles")
            hour = st.slider("Heure du crime", 0, 23, 12, help="Heure de la journée (0-23)")
            day = st.selectbox(
                "Jour de la semaine",
                options=list(DAYS_MAP.keys()),
                format_func=lambda x: DAYS_MAP[x]
            )
            month = st.selectbox(
                "Mois",
                options=list(MONTHS_MAP.keys()),
                format_func=lambda x: MONTHS_MAP[x]
            )
            
            st.markdown("#### 📍 Localisation")
            lat = st.number_input(
                "Latitude",
                min_value=33.0,
                max_value=35.0,
                value=34.0522,
                step=0.0001,
                format="%.4f",
                help="Latitude à Los Angeles (33.0 - 35.0)"
            )
            lon = st.number_input(
                "Longitude",
                min_value=-119.0,
                max_value=-117.0,
                value=-118.2437,
                step=0.0001,
                format="%.4f",
                help="Longitude à Los Angeles (-119.0 à -117.0)"
            )
        
        with col2:
            st.markdown("#### 👤 Informations sur la Victime")
            age = st.number_input(
                "Âge de la victime",
                min_value=0,
                max_value=120,
                value=35,
                help="Âge de la victime (0-120)"
            )
            sex = st.selectbox(
                "Sexe",
                options=["", "M", "F", "X"],
                format_func=lambda x: {
                    "": "Non spécifié",
                    "M": "Masculin",
                    "F": "Féminin",
                    "X": "Autre"
                }.get(x, x)
            )
            descent = st.selectbox(
                "Origine",
                options=["", "H", "B", "W", "A", "O"],
                format_func=lambda x: {
                    "": "Non spécifié",
                    "H": "Hispanique",
                    "B": "Black",
                    "W": "White",
                    "A": "Asian",
                    "O": "Other"
                }.get(x, x)
            )
            
            st.markdown("#### 🏢 Autres Informations")
            area = st.slider(
                "Zone LAPD",
                min_value=1,
                max_value=21,
                value=12,
                help="Zone de police de Los Angeles (1-21)"
            )
            premis = st.number_input(
                "Code du lieu",
                value=101.0,
                help="Code du type de lieu"
            )
            part = st.selectbox("Part 1-2", options=[1, 2])
        
        submitted = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)
    
    if submitted:
        # Préparer le payload
        payload = {
            "Hour": int(hour),
            "Day_of_week": int(day),
            "Month_num": int(month),
            "LAT": float(lat),
            "LON": float(lon),
            "Vict_Age": float(age),
            "AREA": int(area),
            "Vict_Sex": sex if sex else None,
            "Vict_Descent": descent if descent else None,
            "Premis_Cd": float(premis),
            "Part_1_2": int(part)
        }
        
        with st.spinner("🔮 Prédiction en cours..."):
            result = predict(payload)
        
        if result:
            crime = result['predicted_crime_group']
            color = CRIME_COLORS.get(crime, "#333")
            emoji = CRIME_EMOJIS.get(crime, "📋")
            
            # Affichage du résultat
            st.markdown("---")
            st.markdown(f"<h1 style='text-align:center; color:{color}'>{emoji} {crime.upper()}</h1>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Confiance", f"{result['confidence']:.1%}")
            
            with col2:
                st.metric("Code Classe", result['predicted_class_code'])
            
            with col3:
                st.metric("Modèle", f"v{result['model_version']}")
            
            # Détails supplémentaires
            with st.expander("📊 Détails de la prédiction"):
                st.json(result)


# ============================================================================
#                           PAGE : BATCH
# ============================================================================

elif page == "📊 Batch":
    st.title("📊 Prédiction par Batch")
    st.markdown("Téléchargez un fichier CSV pour prédire plusieurs crimes en une fois")
    
    st.info("""
    **Format du fichier CSV requis:**
    
    Colonnes obligatoires : `Hour`, `Day_of_week`, `Month_num`, `LAT`, `LON`
    
    Colonnes optionnelles : `Vict_Age`, `AREA`, `Vict_Sex`, `Vict_Descent`, `Premis_Cd`, `Part_1_2`
    """)
    
    # Télécharger un exemple
    if st.button("📥 Télécharger un exemple de CSV"):
        example_df = pd.DataFrame({
            'Hour': [12, 18, 23],
            'Day_of_week': [0, 5, 6],
            'Month_num': [1, 6, 12],
            'LAT': [34.05, 34.10, 34.15],
            'LON': [-118.24, -118.30, -118.35],
            'Vict_Age': [25, 35, 45],
            'AREA': [10, 15, 20]
        })
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="💾 Télécharger",
            data=csv,
            file_name="example_crimes.csv",
            mime="text/csv"
        )
    
    uploaded = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        st.markdown("### 📄 Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"📊 {len(df)} lignes chargées")
        
        if st.button("🚀 Lancer les prédictions", type="primary"):
            with st.spinner(f"Prédiction de {len(df)} crimes..."):
                try:
                    # Convertir en format JSON
                    records = df.to_dict(orient="records")
                    
                    # Appeler l'API
                    r = requests.post(f"{API_URL}/predict/batch", json=records, timeout=60)
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        st.success(f"✅ {result['total_predictions']} prédictions réussies en {result['processing_time_seconds']:.2f}s")
                        
                        # Créer un DataFrame avec les résultats
                        predictions = result['predictions']
                        results_df = pd.DataFrame([
                            {
                                'Crime_Group': p['predicted_crime_group'],
                                'Class_Code': p['predicted_class_code'],
                                'Confidence': p['confidence']
                            }
                            for p in predictions
                        ])
                        
                        # Combiner avec les données originales
                        final_df = pd.concat([df, results_df], axis=1)
                        
                        # Affichage
                        st.markdown("### 📊 Résultats")
                        st.dataframe(final_df, use_container_width=True)
                        
                        # Télécharger les résultats
                        csv_result = final_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Télécharger les résultats",
                            data=csv_result,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisation
                        st.markdown("### 📈 Distribution des prédictions")
                        fig = px.pie(
                            results_df,
                            names='Crime_Group',
                            title="Répartition des types de crimes",
                            color='Crime_Group',
                            color_discrete_map=CRIME_COLORS
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Erreur : {r.status_code} - {r.text}")
                
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")


# ============================================================================
#                           PAGE : STATISTIQUES
# ============================================================================

elif page == "📈 Statistiques":
    st.title("📈 Statistiques du Système")
    
    health = get_health()
    model_info = get_model_info()
    
    if not health or not model_info:
        st.warning("⚠️ Impossible de récupérer les statistiques")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Prédictions Totales",
                health.get('total_predictions', 0),
                delta="+1" if health.get('total_predictions', 0) > 0 else None
            )
        
        with col2:
            st.metric(
                "Version Modèle",
                f"v{model_info['version']}",
                delta="Production"
            )
        
        with col3:
            accuracy = model_info['metrics'].get('test_accuracy', 0)
            st.metric(
                "Accuracy",
                f"{accuracy:.1%}",
                delta=f"+{(accuracy-0.5)*100:.1f}%"
            )
        
        st.markdown("---")
        
        # Métriques du modèle
        st.markdown("### 📊 Métriques du Modèle")
        
        metrics_df = pd.DataFrame({
            'Métrique': ['Test Accuracy', 'Test F1-Score', 'CV Mean'],
            'Valeur': [
                model_info['metrics'].get('test_accuracy', 0),
                model_info['metrics'].get('test_f1_weighted', 0),
                model_info['metrics'].get('cv_accuracy_mean', 0)
            ]
        })
        
        fig = px.bar(
            metrics_df,
            x='Métrique',
            y='Valeur',
            title="Performance du Modèle",
            color='Valeur',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Classes prédites
        st.markdown("### 🎯 Classes de Crimes")
        
        classes_df = pd.DataFrame({
            'Crime_Group': model_info['classes'],
            'Count': [1] * len(model_info['classes'])  # Placeholder
        })
        
        fig = px.pie(
            classes_df,
            names='Crime_Group',
            title="Types de Crimes Prédits",
            color='Crime_Group',
            color_discrete_map=CRIME_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
#                           FOOTER
# ============================================================================

st.markdown("---")
st.caption("🚀 Projet MLOps 2025 - Pipeline Automatisé | MLflow + FastAPI + Streamlit + GitHub Actions")

# Bouton de refresh global
if st.button("🔄 Rafraîchir les données", key="global_refresh"):
    st.cache_data.clear()
    st.rerun()
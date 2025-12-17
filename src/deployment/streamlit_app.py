import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuration de la page
st.set_page_config(
    page_title="Crime Prediction LA",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API
API_URL = "http://localhost:8000"

# Mapping des jours et mois
DAYS_MAP = {
    0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi",
    4: "Vendredi", 5: "Samedi", 6: "Dimanche"
}

MONTHS_MAP = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}


# ============================================================================
#                           FONCTIONS UTILITAIRES
# ============================================================================

def check_api_health():
    """Vérifie que l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Récupère les infos du modèle"""
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_metrics():
    """Récupère les métriques de production"""
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def predict_crime(features):
    """Appelle l'API pour une prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API : {response.status_code}")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
    return None


# ============================================================================
#                           SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🚨 Crime Prediction LA")
    st.markdown("---")
    
    # Vérifier la santé de l'API
    if check_api_health():
        st.success("✅ API connectée")
    else:
        st.error("❌ API déconnectée")
        st.info("Lancez l'API avec : `uvicorn src.deployment.api:app --reload`")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🎯 Prédiction Simple", "📊 Prédiction Batch", 
         "📈 Statistiques", "🔍 Monitoring", "⚙️ Admin"]
    )
    
    st.markdown("---")
    
    # Info modèle
    st.subheader("📦 Modèle Actuel")
    model_info = get_model_info()
    if model_info:
        st.metric("Version", model_info['model_version'])
        st.metric("Accuracy", f"{model_info['metrics']['test_accuracy']:.2%}")
        st.metric("F1-Score", f"{model_info['metrics']['test_f1']:.2%}")


# ============================================================================
#                           PAGE ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    st.title("🚨 Système de Prédiction des Crimes de Los Angeles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎯 Prédiction Simple**\nFormulaire interactif pour prédire un crime")
    
    with col2:
        st.info("**📊 Prédiction Batch**\nTraiter plusieurs prédictions via CSV")
    
    with col3:
        st.info("**📈 Statistiques**\nVisualiser les tendances et analyses")
    
    st.markdown("---")
    
    st.header("À Propos du Projet")
    
    st.markdown("""
    ### 🎯 Objectif
    Prédire le type de crime à Los Angeles en fonction des caractéristiques temporelles et spatiales.
    
    ### 📊 Modèle
    - **Type** : Ensemble Learning (Stacking/Voting)
    - **Classes prédites** :
        1. 🔴 Violent Crime
        2. 🏠 Property & Theft Crime
        3. 🚗 Vehicle-Related Crime
        4. 📋 Other / Fraud / Public Order Crime
    
    ### 🛠️ Technologies
    - **ML** : Scikit-learn, XGBoost, LightGBM
    - **MLOps** : MLflow, DVC, DagsHub
    - **Backend** : FastAPI
    - **Frontend** : Streamlit
    - **Monitoring** : DeepChecks
    """)
    
    # Métriques globales
    metrics = get_metrics()
    if metrics and metrics.get('total_predictions', 0) > 0:
        st.markdown("---")
        st.header("📊 Statistiques Globales")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prédictions", metrics['total_predictions'])
        col2.metric("Prédictions Récentes", metrics['recent_predictions'])
        col3.metric("Confiance Moyenne", f"{metrics['average_confidence']:.2%}")


# ============================================================================
#                           PAGE PRÉDICTION SIMPLE
# ============================================================================

elif page == "🎯 Prédiction Simple":
    st.title("🎯 Prédiction de Crime")
    
    st.markdown("Remplissez les informations ci-dessous pour prédire le type de crime.")
    
    # Formulaire
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Informations Temporelles")
            hour = st.slider("Heure", 0, 23, 12)
            day_of_week = st.selectbox("Jour de la semaine", options=list(DAYS_MAP.keys()), 
                                       format_func=lambda x: DAYS_MAP[x])
            month = st.selectbox("Mois", options=list(MONTHS_MAP.keys()), 
                                format_func=lambda x: MONTHS_MAP[x])
        
        with col2:
            st.subheader("📍 Informations Spatiales")
            lat = st.number_input("Latitude", min_value=33.0, max_value=35.0, value=34.0522, step=0.0001, format="%.4f")
            lon = st.number_input("Longitude", min_value=-119.0, max_value=-117.0, value=-118.2437, step=0.0001, format="%.4f")
            
            st.subheader("👤 Informations Victimes (Optionnel)")
            vict_age = st.number_input("Âge de la victime", min_value=0, max_value=120, value=30, step=1)
            area = st.number_input("Zone (1-21)", min_value=1, max_value=21, value=1, step=1)
        
        submitted = st.form_submit_button("🔮 Prédire", use_container_width=True)
    
    if submitted:
        # Préparer les features
        features = {
            "Hour": hour,
            "Day_of_week": day_of_week,
            "Month_num": month,
            "LAT": lat,
            "LON": lon,
            "Vict_Age": float(vict_age),
            "AREA": area
        }
        
        # Prédiction
        with st.spinner("🔮 Prédiction en cours..."):
            result = predict_crime(features)
        
        if result:
            st.success("✅ Prédiction réussie !")
            
            # Afficher le résultat
            st.markdown("---")
            st.header("🎯 Résultat")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Type de Crime Prédit", result['predicted_crime'])
            
            with col2:
                st.metric("Confiance", f"{result['confidence']:.2%}")
            
            with col3:
                st.metric("Modèle Version", result['model_version'])
            
            # Détails
            with st.expander("📋 Détails de la Prédiction"):
                st.json(result)


# ============================================================================
#                           PAGE PRÉDICTION BATCH
# ============================================================================

elif page == "📊 Prédiction Batch":
    st.title("📊 Prédiction Batch")
    
    st.markdown("Uploadez un fichier CSV pour prédire plusieurs crimes en une fois.")
    
    # Template CSV
    with st.expander("📄 Télécharger un Template CSV"):
        template_df = pd.DataFrame({
            'Hour': [20, 14, 6],
            'Day_of_week': [5, 2, 1],
            'Month_num': [7, 3, 11],
            'LAT': [34.0522, 34.0522, 34.0522],
            'LON': [-118.2437, -118.2437, -118.2437],
            'Vict Age': [35, 28, 42],
            'AREA': [12, 5, 8]
        })
        
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Télécharger Template",
            data=csv,
            file_name="template_predictions.csv",
            mime="text/csv"
        )
    
    # Upload fichier
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Lire le CSV
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Aperçu des Données")
        st.dataframe(df.head(10))
        
        st.info(f"**{len(df)}** lignes chargées")
        
        if st.button("🚀 Lancer les Prédictions", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            predictions = []
            
            for i, row in df.iterrows():
                features = row.to_dict()
                result = predict_crime(features)
                
                if result:
                    predictions.append({
                        **features,
                        'Predicted_Crime': result['predicted_crime'],
                        'Confidence': result['confidence']
                    })
                
                # Mise à jour progress
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Traitement : {i+1}/{len(df)}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Résultats
            if predictions:
                st.success(f"✅ {len(predictions)} prédictions réussies !")
                
                results_df = pd.DataFrame(predictions)
                
                st.subheader("📊 Résultats")
                st.dataframe(results_df)
                
                # Télécharger résultats
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Télécharger les Résultats",
                    data=csv_results,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualisations
                st.subheader("📈 Visualisations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution des prédictions
                    crime_dist = results_df['Predicted_Crime'].value_counts()
                    fig1 = px.pie(
                        values=crime_dist.values,
                        names=crime_dist.index,
                        title="Distribution des Types de Crimes Prédits"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Confiance moyenne
                    fig2 = px.histogram(
                        results_df,
                        x='Confidence',
                        nbins=20,
                        title="Distribution de la Confiance"
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
#                           PAGE STATISTIQUES
# ============================================================================

elif page == "📈 Statistiques":
    st.title("📈 Statistiques et Analyses")
    
    metrics = get_metrics()
    
    if metrics and metrics.get('total_predictions', 0) > 0:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Prédictions", metrics['total_predictions'])
        col2.metric("Prédictions Récentes", metrics['recent_predictions'])
        col3.metric("Confiance Moyenne", f"{metrics['average_confidence']:.2%}")
        col4.metric("Dernière Prédiction", 
                   datetime.fromisoformat(metrics['last_prediction_time']).strftime("%H:%M:%S"))
        
        st.markdown("---")
        
        # Distribution des crimes
        st.subheader("🎯 Distribution des Types de Crimes")
        
        crime_dist = pd.DataFrame(
            list(metrics['crime_distribution'].items()),
            columns=['Type de Crime', 'Nombre']
        )
        
        fig = px.bar(
            crime_dist,
            x='Type de Crime',
            y='Nombre',
            color='Nombre',
            color_continuous_scale='Viridis',
            title="Distribution des Prédictions par Type de Crime"
        )
        fig.update_xaxis(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 Aucune statistique disponible. Effectuez des prédictions pour voir les analyses.")


# ============================================================================
#                           PAGE MONITORING
# ============================================================================

elif page == "🔍 Monitoring":
    st.title("🔍 Monitoring en Temps Réel")
    
    # Auto-refresh
    if st.checkbox("🔄 Auto-refresh (5s)"):
        import time
        time.sleep(5)
        st.rerun()
    
    # Santé de l'API
    col1, col2 = st.columns(2)
    
    with col1:
        if check_api_health():
            st.success("✅ API opérationnelle")
        else:
            st.error("❌ API indisponible")
    
    with col2:
        model_info = get_model_info()
        if model_info:
            st.info(f"📦 Modèle v{model_info['model_version']} en production")
    
    st.markdown("---")
    
    # Métriques
    metrics = get_metrics()
    if metrics:
        st.subheader("📊 Métriques de Production")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prédictions", metrics['total_predictions'])
        col2.metric("Confiance Moyenne", f"{metrics['average_confidence']:.2%}")
        col3.metric("Prédictions Récentes", metrics['recent_predictions'])
    
    st.markdown("---")
    
    # Info modèle détaillée
    if model_info:
        st.subheader("🤖 Informations du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Nom": model_info['model_name'],
                "Version": model_info['model_version'],
                "Stage": model_info['model_stage'],
                "Nombre de Classes": model_info['n_classes']
            })
        
        with col2:
            st.json({
                "Métriques": model_info['metrics']
            })


# ============================================================================
#                           PAGE ADMIN
# ============================================================================

elif page == "⚙️ Admin":
    st.title("⚙️ Administration")
    
    st.warning("⚠️ Section réservée aux administrateurs")
    
    # Recharger le modèle
    st.subheader("🔄 Gestion du Modèle")
    
    if st.button("🔄 Recharger le Modèle depuis MLflow"):
        try:
            response = requests.post(f"{API_URL}/reload-model")
            if response.status_code == 200:
                st.success("✅ Rechargement du modèle lancé")
            else:
                st.error("❌ Erreur lors du rechargement")
        except:
            st.error("❌ Impossible de contacter l'API")
    
    st.markdown("---")
    
    # Informations système
    st.subheader("💻 Informations Système")
    
    model_info = get_model_info()
    if model_info:
        st.json(model_info)


# ============================================================================
#                           FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚨 Crime Prediction LA | Powered by MLflow, FastAPI & Streamlit</p>
        <p>📊 DagsHub: <a href='https://dagshub.com/benrhoumamohamed752/ProjetMLOps'>Projet MLOps</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
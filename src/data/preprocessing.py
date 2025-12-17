"""
Module de preprocessing pour les données de criminalité de LA
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


def load_data(filepath):
    """Charge les données criminelles depuis un fichier CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Données chargées avec succès. Dimensions : {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return None


def clean_data(df):
    """Nettoie et prépare les données pour l'analyse"""
    print("\n🔧 Nettoyage des données...")
    
    # Suppression des colonnes inutiles
    cols_to_drop = ['Weapon Used Cd', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 
                   'Cross Street', 'Mocodes', 'Weapon Desc']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # Conversion des dates
    date_cols = ['Date Rptd', 'DATE OCC']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Gestion des valeurs manquantes
    # Colonnes catégorielles : remplissage par le mode
    cat_cols = ['Vict Sex', 'Vict Descent', 'Premis Desc', 'Status Desc']
    for col in cat_cols:
        if col in df.columns and len(df[col].mode()) > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Colonnes numériques : remplissage par la médiane
    num_cols = ['Crm Cd 1', 'Premis Cd', 'Vict Age']
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Correction spécifique pour l'âge des victimes
    if 'Vict Age' in df.columns:
        df['Vict Age'] = df['Vict Age'].where(df['Vict Age'] >= 0, df['Vict Age'].median())
    
    # Gestion des coordonnées géographiques
    if 'LON' in df.columns:
        df['LON'] = df['LON'].replace(0.00, pd.NA)
    if 'LAT' in df.columns:
        df['LAT'] = df['LAT'].where(df['LAT'] >= 0, pd.NA)
    
    # Suppression des doublons
    initial_shape = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"   - {initial_shape - df.shape[0]} doublons supprimés")
    
    # Création de caractéristiques temporelles
    if 'DATE OCC' in df.columns:
        df['Day of Week'] = df['DATE OCC'].dt.day_name()
        df['Month'] = df['DATE OCC'].dt.month_name()
        df['Year'] = df['DATE OCC'].dt.year
        df['Day Type'] = df['DATE OCC'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        df['Day_of_week'] = df['DATE OCC'].dt.dayofweek
        df['Month_num'] = df['DATE OCC'].dt.month
    
    if 'TIME OCC' in df.columns:
        df['Hour'] = df['TIME OCC'] // 100
        df['Hour of Day'] = df['Hour']  # Alias pour compatibilité
    
    print(f"✅ Nettoyage terminé. Nouvelle dimension : {df.shape}")
    return df


def prepare_for_ml(df, target_col='Crm Cd Desc', save_encoders=True, encoders_path='model_artifacts'):
    """
    Prépare les données pour le machine learning
    
    Args:
        df: DataFrame nettoyé
        target_col: Colonne cible à prédire
        save_encoders: Si True, sauvegarde les encodeurs
        encoders_path: Chemin pour sauvegarder les encodeurs
    
    Returns:
        X, y, label_encoder
    """
    print("\n🎯 Préparation pour le ML...")
    
    # Supprimer les lignes avec des valeurs manquantes dans les features importantes
    required_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', target_col]
    df_ml = df.dropna(subset=required_cols).copy()
    
    print(f"   - Données après suppression des NaN : {df_ml.shape}")
    
    # Encodage de la variable cible
    le = LabelEncoder()
    y = le.fit_transform(df_ml[target_col])
    
    # Sélection des features
    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON']
    
    # Optionnel : Ajouter d'autres features si disponibles
    optional_features = ['Vict Age', 'AREA']
    for feat in optional_features:
        if feat in df_ml.columns:
            if df_ml[feat].isna().sum() / len(df_ml) < 0.3:  # Moins de 30% de NaN
                df_ml[feat].fillna(df_ml[feat].median(), inplace=True)
                feature_cols.append(feat)
    
    X = df_ml[feature_cols].copy()
    
    # Sauvegarder les encodeurs si demandé
    if save_encoders:
        os.makedirs(encoders_path, exist_ok=True)
        joblib.dump(le, os.path.join(encoders_path, 'label_encoder.joblib'))
        
        # Sauvegarder aussi les noms des features
        feature_info = {
            'features': feature_cols,
            'target': target_col,
            'classes': le.classes_.tolist()
        }
        joblib.dump(feature_info, os.path.join(encoders_path, 'feature_info.joblib'))
        print(f"✅ Encodeurs sauvegardés dans {encoders_path}")
    
    print(f"   - Features utilisées : {feature_cols}")
    print(f"   - Nombre de classes : {len(le.classes_)}")
    print(f"   - Forme finale X : {X.shape}, y : {y.shape}")
    
    return X, y, le


def filter_by_year(df, year):
    """Filtre les données par année"""
    if 'Year' not in df.columns:
        if 'DATE OCC' in df.columns:
            df['Year'] = pd.to_datetime(df['DATE OCC']).dt.year
        else:
            raise ValueError("Impossible de déterminer l'année")
    
    df_filtered = df[df['Year'] == year].copy()
    print(f"✅ Données filtrées pour l'année {year} : {df_filtered.shape}")
    return df_filtered


def get_data_summary(df):
    """Affiche un résumé des données"""
    print("\n📊 Résumé des données :")
    print(f"   - Forme : {df.shape}")
    print(f"   - Colonnes : {df.columns.tolist()}")
    print(f"\n   - Valeurs manquantes :")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("      Aucune valeur manquante !")
    
    if 'Crm Cd Desc' in df.columns:
        print(f"\n   - Types de crimes : {df['Crm Cd Desc'].nunique()}")
        print(f"   - Top 5 des crimes :")
        print(df['Crm Cd Desc'].value_counts().head())


if __name__ == "__main__":
    # Test du module
    filepath = "data/raw/crime_Data_2020.csv"
    
    # Charger et nettoyer
    df = load_data(filepath)
    if df is not None:
        df_clean = clean_data(df)
        
        # Filtrer pour 2020
        df_2020 = filter_by_year(df_clean, 2020)
        
        # Résumé
        get_data_summary(df_2020)
        
        # Préparer pour ML
        X, y, le = prepare_for_ml(df_2020)
        
        # Sauvegarder
        output_path = 'data/processed/crime_2020_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_2020.to_csv(output_path, index=False)
        print(f"\n✅ Données sauvegardées : {output_path}")

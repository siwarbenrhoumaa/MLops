"""
Module de preprocessing pour les données de criminalité de LA
Version corrigée et optimisée pour Deepchecks + modélisation (2020)
→ Crm Cd supprimé pour éviter le data leakage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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


# ---------------------------------------------------------
# Fonction de regroupement en 4 classes (identique à train.py)
# ---------------------------------------------------------
def map_crime_group_4(desc):
    if pd.isna(desc):
        return "Other / Fraud / Public Order Crime"
    desc = str(desc).upper()
    if any(k in desc for k in ["ASSAULT", "BATTERY", "ROBBERY", "HOMICIDE", "RAPE", "SEX", "SODOMY", "LEWD"]):
        return "Violent Crime"
    if any(k in desc for k in ["THEFT", "BURGLARY", "SHOPLIFTING", "VANDALISM", "ARSON"]):
        return "Property & Theft Crime"
    if any(k in desc for k in ["VEHICLE", "DWOC", "MOTOR VEHICLE", "BOAT"]):
        return "Vehicle-Related Crime"
    return "Other / Fraud / Public Order Crime"


def clean_data(df):
    """Nettoie et prépare les données pour l'analyse"""
    print("\n🔧 Nettoyage des données...")

    # === SAUVEGARDE D'UNE VERSION COMPLÈTE AVANT NETTOYAGE AGRESSIF ===
    df_full = df.copy()

    # Suppression des colonnes inutiles au début
    cols_to_drop_initial = ['Weapon Used Cd', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4',
                            'Cross Street', 'Mocodes', 'Weapon Desc']
    existing_initial = [col for col in cols_to_drop_initial if col in df.columns]
    df = df.drop(columns=existing_initial, axis=1)
    print(f"   - Colonnes supprimées (phase initiale) : {existing_initial}")

    # Conversion des dates
    date_cols = ['Date Rptd', 'DATE OCC']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

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
        df['Hour of Day'] = df['Hour']

    # === CRÉATION DE Crime_Group (avant suppression de Crm Cd Desc) ===
    if 'Crm Cd Desc' in df.columns:
        df['Crime_Group'] = df['Crm Cd Desc'].apply(map_crime_group_4)
        print(f"   - Crime_Group créé : {df['Crime_Group'].nunique()} classes (Violent, Property, Vehicle, Other)")

    # === NETTOYAGE AGRESSIF POUR DEEPCHECKS ET ÉVITER DATA LEAKAGE ===
    print("\n   🔹 Nettoyage agressif pour qualité Deepchecks et éviter data leakage...")
    cols_to_drop_aggressive = [
        'DR_NO', 'Rpt Dist No',                    # IDs inutiles
        'AREA NAME', 'Premis Desc', 'Status Desc', # Descriptions redondantes
        'LOCATION',                                # Source de String Mismatch
        'Crm Cd Desc',                             # Trop de classes → remplacé par Crime_Group
        'Crm Cd',                                  # ← SUPPRIMÉ ICI : cause directe de leakage !
        'Status',                                  # Peu prédictif
        'Date Rptd', 'TIME OCC', 'DATE OCC'        # Remplacés par features extraites
    ]
    existing_aggressive = [col for col in cols_to_drop_aggressive if col in df.columns]
    df = df.drop(columns=existing_aggressive, axis=1)
    print(f"   - Colonnes supprimées (réduction bruit/corrélations/leakage) : {existing_aggressive}")

    # Gestion des valeurs manquantes
    cat_cols = ['Vict Sex', 'Vict Descent']
    for col in cat_cols:
        if col in df.columns and len(df[col].mode()) > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    num_cols = ['Crm Cd 1', 'Premis Cd', 'Vict Age', 'AREA']
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Correction des âges aberrants
    if 'Vict Age' in df.columns:
        median_age = df['Vict Age'].median()
        df['Vict Age'] = df['Vict Age'].where((df['Vict Age'] >= 0) & (df['Vict Age'] <= 120), median_age)
        print(f"   - Âges aberrants corrigés (remplacés par médiane : {median_age})")

    # === FILTRAGE GPS STRICT (élimine les 791 lignes invalides) ===
    initial_rows = len(df)
    if {'LAT', 'LON'}.issubset(df.columns):
        df = df[~((df['LAT'] == 0) & (df['LON'] == 0))]  # Cas classique LAPD
        df = df[df['LAT'].between(33.7, 34.4)]
        df = df[df['LON'].between(-118.7, -118.1)]
        removed = initial_rows - len(df)
        print(f"   - {removed} lignes GPS invalides supprimées (hors zone Los Angeles)")

    # Suppression des doublons
    initial_shape = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"   - {initial_shape - df.shape[0]} doublons supprimés")

    print(f"✅ Nettoyage terminé. Nouvelle dimension : {df.shape}")

    # Retourner les deux versions
    return df, df_full  # df = version propre, df_full = version complète


def prepare_for_ml(df, target_col='Crime_Group', save_encoders=True, encoders_path='model_artifacts'):
    """
    Prépare les données pour le machine learning
    Utilise Crime_Group comme cible (4 classes)
    """
    print("\n🎯 Préparation pour le ML...")

    required_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', target_col]
    df_ml = df.dropna(subset=required_cols).copy()
    print(f"   - Données après suppression des NaN critiques : {df_ml.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df_ml[target_col])

    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']

    optional_features = ['Vict Sex', 'Vict Descent', 'Premis Cd']
    for feat in optional_features:
        if feat in df_ml.columns:
            missing_ratio = df_ml[feat].isna().sum() / len(df_ml)
            if missing_ratio < 0.3:
                df_ml[feat].fillna(df_ml[feat].mode()[0] if df_ml[feat].dtype == 'object' else df_ml[feat].median(), inplace=True)
                feature_cols.append(feat)

    X = df_ml[feature_cols].copy()

    if save_encoders:
        os.makedirs(encoders_path, exist_ok=True)
        joblib.dump(le, os.path.join(encoders_path, 'label_encoder.joblib'))
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
            df['Year'] = pd.to_datetime(df['DATE OCC'], errors='coerce').dt.year
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
        print("      Aucune valeur manquante critique !")

    if 'Crime_Group' in df.columns:
        print(f"\n   - Groupes de crimes : {df['Crime_Group'].nunique()}")
        print(f"   - Distribution :")
        print(df['Crime_Group'].value_counts())


if __name__ == "__main__":
    # Test du module
    filepath = "data/raw/crime_Data_2020.csv"

    df = load_data(filepath)
    if df is not None:
        # Nettoyage retourne deux versions
        df_clean, df_full = clean_data(df)

        # Filtrer pour 2020
        df_2020_clean = filter_by_year(df_clean, 2020)
        df_2020_full = filter_by_year(df_full, 2020)

        # Résumé
        get_data_summary(df_2020_clean)

        # Préparer pour ML
        X, y, le = prepare_for_ml(df_2020_clean, target_col='Crime_Group')

        # Sauvegarde des deux versions
        output_clean = 'data/processed/crime_2020_processed2.csv'
        os.makedirs('data/processed', exist_ok=True)

        df_2020_clean.to_csv(output_clean, index=False)

        print(f"\n✅ Données propres sauvegardées : {output_clean}")
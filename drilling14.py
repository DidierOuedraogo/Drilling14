import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io
import tempfile
import os

# Import conditionnels pour PyVista
try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    st.warning("PyVista n'est pas installé. La visualisation 3D avancée ne sera pas disponible.")

# Configuration de la page
st.set_page_config(
    page_title="Mining Geology Data Application",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4053;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E4053;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 1rem;
        color: #566573;
        font-style: italic;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9F9;
        padding: 10px 15px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2874A6;
        color: white;
    }
    .uploadedFile {
        border: 1px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
    }
    .success-message {
        background-color: #D4EFDF;
        border-left: 5px solid #2ECC71;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .warning-message {
        background-color: #FCF3CF;
        border-left: 5px solid #F1C40F;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .error-message {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .info-card {
        background-color: #EBF5FB;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #F8F9F9;
        border-left: 4px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9F9;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application et auteur
st.markdown('<h1 class="main-header">Mining Geology Data Application</h1>', unsafe_allow_html=True)
st.markdown('<p class="author">Développé par: Didier Ouedraogo, P.Geo.</p>', unsafe_allow_html=True)

# Fonction pour convertir les chaînes en nombres flottants avec gestion d'erreurs
def safe_float(value):
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #3498DB; text-decoration: none;"><button style="background-color: #3498DB; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">Télécharger {text}</button></a>'
    return href

# Fonction pour vérifier si une colonne existe dans un DataFrame
def column_exists(df, column_name):
    return df is not None and column_name and column_name in df.columns

# Fonction pour créer des composites d'analyses avec coordonnées
def create_composites(assays_df, hole_id_col, from_col, to_col, value_col, composite_length=1.0, 
                     collars_df=None, survey_df=None, x_col=None, y_col=None, z_col=None, 
                     azimuth_col=None, dip_col=None, depth_col=None):
    if assays_df is None or assays_df.empty:
        return None
    
    # Vérifier que toutes les colonnes nécessaires existent
    if not all(col in assays_df.columns for col in [hole_id_col, from_col, to_col, value_col]):
        st.markdown('<div class="error-message">Colonnes manquantes dans le DataFrame des analyses</div>', unsafe_allow_html=True)
        return None
    
    # Créer une copie des données pour éviter de modifier l'original
    df = assays_df.copy()
    
    # Convertir les colonnes numériques en flottants
    for col in [from_col, to_col, value_col]:
        df[col] = df[col].apply(safe_float)
    
    # Initialiser le DataFrame des composites
    composites = []
    
    # Pour chaque trou de forage
    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].sort_values(by=from_col)
        
        if hole_data.empty:
            continue

        # Récupérer les données de collars et survey pour les coordonnées si disponibles
        collar_info = None
        surveys = None
        
        if collars_df is not None and survey_df is not None and all(col is not None for col in [x_col, y_col, z_col, depth_col, azimuth_col, dip_col]):
            if hole_id in collars_df[hole_id_col].values:
                collar_info = collars_df[collars_df[hole_id_col] == hole_id].iloc[0]
                surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        # Pour chaque intervalle de composite
        composite_start = float(hole_data[from_col].min())
        while composite_start < float(hole_data[to_col].max()):
            composite_end = composite_start + composite_length
            
            # Trouver tous les intervalles qui chevauchent le composite actuel
            overlapping = hole_data[
                ((hole_data[from_col] >= composite_start) & (hole_data[from_col] < composite_end)) |
                ((hole_data[to_col] > composite_start) & (hole_data[to_col] <= composite_end)) |
                ((hole_data[from_col] <= composite_start) & (hole_data[to_col] >= composite_end))
            ]
            
            if not overlapping.empty:
                # Calculer le poids pondéré pour chaque intervalle chevauchant
                weighted_values = []
                total_length = 0
                
                for _, row in overlapping.iterrows():
                    overlap_start = max(composite_start, row[from_col])
                    overlap_end = min(composite_end, row[to_col])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0:
                        weighted_values.append(row[value_col] * overlap_length)
                        total_length += overlap_length
                
                # Calculer la valeur pondérée du composite
                if total_length > 0:
                    composite_value = sum(weighted_values) / total_length
                    
                    # Créer une entrée de composite de base
                    composite_entry = {
                        hole_id_col: hole_id,
                        'From': composite_start,
                        'To': composite_end,
                        'Length': total_length,
                        value_col: composite_value
                    }
                    
                    # Ajouter les coordonnées si les données nécessaires sont disponibles
                    if collar_info is not None and not surveys.empty:
                        # Calculer la position moyenne (milieu de l'intervalle)
                        mid_depth = (composite_start + composite_end) / 2
                        
                        # Chercher les données de survey les plus proches
                        closest_idx = surveys[depth_col].apply(lambda d: abs(d - mid_depth)).idxmin()
                        closest_survey = surveys.loc[closest_idx]
                        
                        # Récupérer les données du collar
                        x_start = safe_float(collar_info[x_col])
                        y_start = safe_float(collar_info[y_col])
                        z_start = safe_float(collar_info[z_col])
                        
                        # Calculer les coordonnées 3D approximatives pour le composite
                        # (Méthode simplifiée - pour une précision parfaite, une interpolation plus complexe serait nécessaire)
                        depth = safe_float(closest_survey[depth_col])
                        azimuth = safe_float(closest_survey[azimuth_col])
                        dip = safe_float(closest_survey[dip_col])
                        
                        # Convertir l'azimuth et le dip en direction 3D
                        azimuth_rad = np.radians(azimuth)
                        dip_rad = np.radians(dip)
                        
                        # Calculer la position approximative
                        dx = depth * np.sin(dip_rad) * np.sin(azimuth_rad)
                        dy = depth * np.sin(dip_rad) * np.cos(azimuth_rad)
                        dz = -depth * np.cos(dip_rad)  # Z est négatif pour la profondeur
                        
                        # Ajouter les coordonnées au composite
                        composite_entry['X'] = x_start + dx
                        composite_entry['Y'] = y_start + dy
                        composite_entry['Z'] = z_start + dz
                    
                    # Ajouter le composite au résultat
                    composites.append(composite_entry)
            
            composite_start = composite_end
    
    # Créer un DataFrame à partir des composites
    if composites:
        return pd.DataFrame(composites)
    else:
        return pd.DataFrame()

# Fonction pour créer un strip log pour un forage spécifique
def create_strip_log(hole_id, collars_df, survey_df, lithology_df, assays_df, 
                    hole_id_col, depth_col, 
                    lith_from_col, lith_to_col, lith_col,
                    assay_from_col, assay_to_col, assay_value_col):
    
    # Vérifier si les données nécessaires sont disponibles
    if collars_df is None or survey_df is None:
        return None
    
    # Récupérer les informations du forage
    hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
    
    if hole_surveys.empty:
        return None
    
    # Convertir les valeurs de profondeur en flottants
    hole_surveys[depth_col] = hole_surveys[depth_col].apply(safe_float)
    
    # Profondeur maximale du forage
    max_depth = hole_surveys[depth_col].max()
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(12, max_depth/10 + 2), 
                            gridspec_kw={'width_ratios': [2, 1, 3]})
    
    # Titre du graphique
    fig.suptitle(f'Strip Log - Forage {hole_id}', fontsize=16)
    
    # 1. Colonne de lithologie
    if lithology_df is not None and all(col and col in lithology_df.columns for col in [hole_id_col, lith_from_col, lith_to_col, lith_col]):
        hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id].sort_values(by=lith_from_col)
        
        if not hole_litho.empty:
            # Convertir les colonnes de profondeur en flottants
            hole_litho[lith_from_col] = hole_litho[lith_from_col].apply(safe_float)
            hole_litho[lith_to_col] = hole_litho[lith_to_col].apply(safe_float)
            
            # Définir une palette de couleurs pour les différentes lithologies
            unique_liths = hole_litho[lith_col].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_liths)))
            lith_color_map = {lith: color for lith, color in zip(unique_liths, colors)}
            
            # Dessiner des rectangles pour chaque intervalle de lithologie
            for _, row in hole_litho.iterrows():
                lith_from = row[lith_from_col]
                lith_to = row[lith_to_col]
                lith_type = row[lith_col]
                
                axes[0].add_patch(plt.Rectangle((0, lith_from), 1, lith_to - lith_from, 
                                                color=lith_color_map[lith_type]))
                
                # Ajouter le texte de la lithologie au milieu de l'intervalle
                interval_height = lith_to - lith_from
                font_size = min(10, max(6, interval_height * 0.8))  # Taille de police adaptative
                
                axes[0].text(0.5, (lith_from + lith_to) / 2, lith_type,
                            ha='center', va='center', fontsize=font_size,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Configurer l'axe de la lithologie
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(max_depth, 0)  # Inverser l'axe y pour que la profondeur augmente vers le bas
    axes[0].set_xlabel('Lithologie')
    axes[0].set_ylabel('Profondeur (m)')
    axes[0].set_xticks([])
    
    # 2. Colonne de profondeur
    depth_ticks = np.arange(0, max_depth + 10, 10)
    axes[1].set_yticks(depth_ticks)
    axes[1].set_ylim(max_depth, 0)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_xlabel('Profondeur')
    axes[1].grid(axis='y')
    
    # 3. Colonne d'analyses
    if assays_df is not None and all(col and col in assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
        hole_assays = assays_df[assays_df[hole_id_col] == hole_id].sort_values(by=assay_from_col)
        
        if not hole_assays.empty:
            # Convertir les colonnes numériques en flottants
            hole_assays[assay_from_col] = hole_assays[assay_from_col].apply(safe_float)
            hole_assays[assay_to_col] = hole_assays[assay_to_col].apply(safe_float)
            hole_assays[assay_value_col] = hole_assays[assay_value_col].apply(safe_float)
            
            # Trouver la valeur maximale pour normaliser
            max_value = hole_assays[assay_value_col].max()
            
            # Dessiner des barres horizontales pour chaque intervalle d'analyse
            for _, row in hole_assays.iterrows():
                assay_from = row[assay_from_col]
                assay_to = row[assay_to_col]
                assay_value = row[assay_value_col]
                
                # Dessiner une barre horizontale pour la valeur
                bar_width = (assay_value / max_value) * 0.9 if max_value > 0 else 0  # Normaliser la largeur
                axes[2].add_patch(plt.Rectangle((0, assay_from), bar_width, assay_to - assay_from, 
                                                color='red', alpha=0.7))
                
                # Ajouter la valeur comme texte avec taille de police adaptative
                interval_height = assay_to - assay_from
                font_size = min(12, max(7, interval_height * 1))  # Taille de police ajustée pour les teneurs
                
                # Afficher seulement si l'intervalle est assez grand
                if interval_height > 1 or assay_value > max_value * 0.5:  # Afficher les valeurs importantes même dans de petits intervalles
                    axes[2].text(bar_width + 0.05, (assay_from + assay_to) / 2, f"{assay_value:.2f}",
                                va='center', fontsize=font_size, fontweight='bold')
    
    # Configurer l'axe des analyses
    axes[2].set_xlim(0, 1.2)
    axes[2].set_ylim(max_depth, 0)
    axes[2].set_xlabel(f'Analyses ({assay_value_col if assay_value_col else ""})')
    axes[2].grid(axis='y')
    
    plt.tight_layout()
    
    # Convertir le graphique en image pour Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Fonction pour créer une carte 2D des collars
def create_collar_map(collars_df, hole_id_col, x_col, y_col, color_by=None, size_by=None):
    if collars_df is None or collars_df.empty:
        return None
    
    if not all(col in collars_df.columns for col in [hole_id_col, x_col, y_col]):
        st.markdown('<div class="error-message">Colonnes nécessaires manquantes pour la carte des collars</div>', unsafe_allow_html=True)
        return None
    
    # Créer une copie propre des données
    df = collars_df.copy()
    
    # Convertir les coordonnées en flottants
    df[x_col] = df[x_col].apply(safe_float)
    df[y_col] = df[y_col].apply(safe_float)
    
    # Configurer le tracé selon les options
    if color_by and color_by in df.columns:
        # Si la coloration est numérique
        if pd.api.types.is_numeric_dtype(df[color_by]) or df[color_by].apply(lambda x: isinstance(x, (int, float))).all():
            df[color_by] = pd.to_numeric(df[color_by], errors='coerce')
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col, 
                color=color_by,
                color_continuous_scale="Viridis",
                hover_name=hole_id_col,
                size=size_by if size_by and size_by in df.columns else None,
                labels={
                    x_col: "X",
                    y_col: "Y",
                    color_by: color_by,
                    size_by: size_by if size_by else ""
                },
                title="Carte des collars"
            )
        else:
            # Si la coloration est catégorielle
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col, 
                color=color_by,
                hover_name=hole_id_col,
                size=size_by if size_by and size_by in df.columns else None,
                labels={
                    x_col: "X",
                    y_col: "Y",
                    color_by: color_by,
                    size_by: size_by if size_by else ""
                },
                title="Carte des collars"
            )
    else:
        # Sans coloration spéciale
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            hover_name=hole_id_col,
            size=size_by if size_by and size_by in df.columns else None,
            labels={
                x_col: "X",
                y_col: "Y",
                size_by: size_by if size_by else ""
            },
            title="Carte des collars"
        )
    
    # Ajouter les annotations de texte pour les identifiants de forage
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=str(row[hole_id_col]),
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title="X",
            showgrid=True,
            zeroline=True,
            showline=True,
            linecolor="black",
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title="Y",
            showgrid=True,
            zeroline=True,
            showline=True,
            linecolor="black",
            gridcolor="lightgray",
            scaleanchor="x",  # Make sure x and y have same scale
            scaleratio=1      # Ensure 1:1 aspect ratio
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='#3498DB',
            borderwidth=1
        ),
        height=600
    )
    
    return fig

# Fonction pour générer la visualisation 3D avec PyVista
def create_pyvista_3d(collars_df, survey_df, lithology_df=None, assays_df=None, 
                      hole_id_col=None, x_col=None, y_col=None, z_col=None,
                      azimuth_col=None, dip_col=None, depth_col=None,
                      lith_from_col=None, lith_to_col=None, lith_col=None,
                      assay_from_col=None, assay_to_col=None, assay_value_col=None,
                      selected_holes=None, show_lithology=True, show_assays=True):
    
    if not PYVISTA_AVAILABLE:
        return None, "PyVista n'est pas disponible. Impossible de créer la visualisation 3D."
    
    if collars_df is None or survey_df is None:
        return None, "Données de collar ou survey manquantes."
    
    # Sélectionner uniquement les trous demandés
    if selected_holes:
        collars_df = collars_df[collars_df[hole_id_col].isin(selected_holes)]
        survey_df = survey_df[survey_df[hole_id_col].isin(selected_holes)]
        if lithology_df is not None:
            lithology_df = lithology_df[lithology_df[hole_id_col].isin(selected_holes)]
        if assays_df is not None:
            assays_df = assays_df[assays_df[hole_id_col].isin(selected_holes)]
    
    # Vérifier que les colonnes nécessaires existent
    if not all(col and col in collars_df.columns for col in [hole_id_col, x_col, y_col, z_col]) or \
       not all(col and col in survey_df.columns for col in [hole_id_col, azimuth_col, dip_col, depth_col]):
        return None, "Colonnes nécessaires manquantes dans les DataFrames collars ou survey."
    
    # Configurer PyVista pour fonctionner avec Streamlit
    pv.set_jupyter_backend('static')
    
    # Créer une scène PyVista
    p = pv.Plotter(notebook=True, off_screen=True)
    p.set_background('white')
    
    # Couleurs pour les différents trous de forage
    cmap = plt.cm.get_cmap('tab20', len(collars_df[hole_id_col].unique()))
    hole_colors = {hole: cmap(i)[:3] for i, hole in enumerate(collars_df[hole_id_col].unique())}
    
    # Convertir les colonnes numériques en flottants
    for col in [x_col, y_col, z_col]:
        collars_df[col] = collars_df[col].apply(safe_float)
    
    for col in [azimuth_col, dip_col, depth_col]:
        survey_df[col] = survey_df[col].apply(safe_float)
    
    # Dictionnaire pour stocker les valeurs d'analyse
    assay_values = {}
    max_assay_value = 0
    
    if show_assays and assays_df is not None and all(col and col in assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
        # Convertir les valeurs d'analyse en flottants
        assays_df[assay_value_col] = assays_df[assay_value_col].apply(safe_float)
        assays_df[assay_from_col] = assays_df[assay_from_col].apply(safe_float)
        assays_df[assay_to_col] = assays_df[assay_to_col].apply(safe_float)
        
        max_assay_value = assays_df[assay_value_col].max()
    
    # Dictionnaire pour stocker les lithologies
    litho_info = {}
    
    if show_lithology and lithology_df is not None and all(col and col in lithology_df.columns for col in [hole_id_col, lith_from_col, lith_to_col, lith_col]):
        # Convertir les colonnes de profondeur en flottants
        lithology_df[lith_from_col] = lithology_df[lith_from_col].apply(safe_float)
        lithology_df[lith_to_col] = lithology_df[lith_to_col].apply(safe_float)
        
        # Créer une palette de couleurs pour les lithologies
        unique_liths = lithology_df[lith_col].unique()
        lith_cmap = plt.cm.get_cmap('Paired', len(unique_liths))
        lith_colors = {lith: lith_cmap(i)[:3] for i, lith in enumerate(unique_liths)}
    
    # Pour chaque trou de forage
    for hole_id in collars_df[hole_id_col].unique():
        # Récupérer les données de collar pour ce trou
        collar = collars_df[collars_df[hole_id_col] == hole_id]
        if collar.empty:
            continue
            
        # Point de départ du trou (collar)
        x_start = collar[x_col].values[0]
        y_start = collar[y_col].values[0]
        z_start = collar[z_col].values[0]
        
        # Récupérer les données de survey pour ce trou
        hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        if hole_surveys.empty:
            continue
            
        # Calculer les points 3D pour le tracé du trou
        x_points = [x_start]
        y_points = [y_start]
        z_points = [z_start]
        
        depths = [0]  # Commencer à profondeur 0
        
        current_x, current_y, current_z = x_start, y_start, z_start
        prev_depth = 0
        
        for _, survey in hole_surveys.iterrows():
            depth = survey[depth_col]
            azimuth = survey[azimuth_col]
            dip = survey[dip_col]
            
            segment_length = depth - prev_depth
            
            # Convertir l'azimuth et le dip en direction 3D
            azimuth_rad = np.radians(azimuth)
            dip_rad = np.radians(dip)
            
            # Calculer la nouvelle position
            dx = segment_length * np.sin(dip_rad) * np.sin(azimuth_rad)
            dy = segment_length * np.sin(dip_rad) * np.cos(azimuth_rad)
            dz = -segment_length * np.cos(dip_rad)  # Z est négatif pour la profondeur
            
            current_x += dx
            current_y += dy
            current_z += dz
            
            x_points.append(current_x)
            y_points.append(current_y)
            z_points.append(current_z)
            depths.append(depth)
            
            prev_depth = depth
        
        # Créer une ligne pour le forage
        points = np.column_stack((x_points, y_points, z_points))
        line = pv.Line(points)
        
        # Ajouter la ligne à la scène
        p.add_mesh(line, color=hole_colors[hole_id], line_width=5, label=f"Forage {hole_id}")
        
        # Ajouter un marqueur pour le collar
        collar_sphere = pv.Sphere(radius=2, center=(x_start, y_start, z_start))
        p.add_mesh(collar_sphere, color=hole_colors[hole_id])
        p.add_point_labels([(x_start, y_start, z_start)], [f"{hole_id}"], point_size=1, font_size=12)
        
        # Récupérer les lithologies pour ce trou
        if show_lithology and lithology_df is not None and hole_id in lithology_df[hole_id_col].unique():
            hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id]
            
            for _, litho in hole_litho.iterrows():
                from_depth = litho[lith_from_col]
                to_depth = litho[lith_to_col]
                lith_type = litho[lith_col]
                
                # Calculer les points le long du forage pour cette intervalle
                from_idx = np.interp(from_depth, depths, np.arange(len(depths)))
                to_idx = np.interp(to_depth, depths, np.arange(len(depths)))
                
                # Créer une interpolation si on a suffisamment de points
                if len(points) > 1:
                    from_i = int(from_idx)
                    to_i = int(to_idx)
                    
                    # S'assurer que les indices sont dans la plage valide
                    from_i = max(0, min(from_i, len(points) - 1))
                    to_i = max(0, min(to_i, len(points) - 1))
                    
                    # Si l'intervalle de lithologie ne contient qu'un seul point
                    if from_i == to_i:
                        if from_i < len(points) - 1:
                            to_i = from_i + 1
                        else:
                            from_i = to_i - 1
                    
                    # Extraire les points pour cet intervalle
                    litho_points = points[from_i:to_i+1]
                    
                    # Créer un tube pour représenter l'intervalle lithologique
                    if len(litho_points) > 1:
                        litho_line = pv.Line(litho_points)
                        litho_tube = litho_line.tube(radius=1.5)
                        
                        # Ajouter le tube à la scène avec la couleur de la lithologie
                        p.add_mesh(litho_tube, color=lith_colors[lith_type], 
                                  label=f"{hole_id} - {lith_type}" if f"{hole_id} - {lith_type}" not in litho_info else None)
                        
                        # Stocker l'info pour la légende
                        litho_info[f"{hole_id} - {lith_type}"] = lith_colors[lith_type]
        
        # Récupérer les analyses pour ce trou
        if show_assays and assays_df is not None and hole_id in assays_df[hole_id_col].unique():
            hole_assays = assays_df[assays_df[hole_id_col] == hole_id]
            
            for _, assay in hole_assays.iterrows():
                from_depth = assay[assay_from_col]
                to_depth = assay[assay_to_col]
                value = assay[assay_value_col]
                
                # Normaliser la valeur pour l'échelle de couleur et la taille
                normalized_value = value / max_assay_value if max_assay_value > 0 else 0
                sphere_radius = 1 + 4 * normalized_value  # Rayon entre 1 et 5
                
                # Calculer le point médian pour cette analyse
                mid_depth = (from_depth + to_depth) / 2
                mid_idx = np.interp(mid_depth, depths, np.arange(len(depths)))
                mid_i = int(mid_idx)
                mid_i = max(0, min(mid_i, len(points) - 1))
                
                if mid_i < len(points):
                    # Créer une sphère à cet emplacement
                    x, y, z = points[mid_i]
                    
                    # Appliquer un petit décalage pour que la sphère soit visible à côté du forage
                    offset = 2
                    sphere = pv.Sphere(radius=sphere_radius, center=(x + offset, y, z))
                    
                    # Créer une couleur basée sur la valeur (rouge pour valeurs élevées)
                    color = [normalized_value, 0, 0]  # Teinte rouge basée sur la valeur
                    
                    # Ajouter la sphère à la scène
                    p.add_mesh(sphere, color=color)
                    
                    # Ajouter une étiquette de texte avec la valeur
                    p.add_point_labels([(x + offset, y, z)], [f"{value:.2f}"], point_size=1, font_size=10)
                    
                    # Stocker la valeur pour la légende
                    assay_values[value] = color
    
    # Ajouter une légende
    p.add_legend(bcolor=(1, 1, 1), face="rectangle")
    
    # Ajouter un titre
    p.add_title("Visualisation 3D des forages", font_size=18)
    
    # Ajouter des axes
    p.show_axes()
    
    # Montrer les contours du modèle
    p.show_bounds(grid='front', location='outer')
    
    # Générer l'image
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    p.screenshot(temp_file.name, window_size=(1200, 800), return_img=False)
    
    # Fermer le plotter pour libérer la mémoire
    p.close()
    
    # Chemin vers l'image générée
    image_path = temp_file.name
    
    return image_path, None

# Fonction pour filtrer un DataFrame en fonction de critères
def filter_dataframe(df, filter_column, filter_type, filter_value):
    """
    Filtre un DataFrame selon un critère spécifié.
    
    Args:
        df: DataFrame à filtrer
        filter_column: Colonne à utiliser pour le filtrage
        filter_type: Type de filtre ('equal', 'greater', 'less', 'contains')
        filter_value: Valeur pour le filtrage
    
    Returns:
        DataFrame filtré
    """
    if df is None or df.empty or filter_column not in df.columns:
        return df
    
    # Détecter le type de la colonne
    col_type = df[filter_column].dtype
    
    # Filtrer selon le type de filtre
    if filter_type == 'equal':
        if pd.api.types.is_numeric_dtype(col_type):
            try:
                filter_value = float(filter_value)
                return df[df[filter_column] == filter_value]
            except (ValueError, TypeError):
                return df
        else:
            return df[df[filter_column] == filter_value]
    
    elif filter_type == 'greater':
        if pd.api.types.is_numeric_dtype(col_type):
            try:
                filter_value = float(filter_value)
                return df[df[filter_column] > filter_value]
            except (ValueError, TypeError):
                return df
        return df
    
    elif filter_type == 'less':
        if pd.api.types.is_numeric_dtype(col_type):
            try:
                filter_value = float(filter_value)
                return df[df[filter_column] < filter_value]
            except (ValueError, TypeError):
                return df
        return df
    
    elif filter_type == 'contains':
        if pd.api.types.is_string_dtype(col_type) or df[filter_column].apply(lambda x: isinstance(x, str)).all():
            return df[df[filter_column].str.contains(str(filter_value), case=False, na=False)]
        return df
    
    return df

# Barre latérale pour la navigation
with st.sidebar:
    st.markdown('<h2 style="color: #3498DB;">Navigation</h2>', unsafe_allow_html=True)
    page = st.radio('', [
        'Chargement des données', 
        'Aperçu des données', 
        'Composites',
        'Carte des Collars', 
        'Strip Logs',
        'Visualisation 3D'
    ], index=0, key='navigation')
    
    st.markdown('---')
    st.markdown('<div style="padding: 20px; background-color: #EBF5FB; border-radius: 5px;">', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-weight: bold;">Mining Geology Data Application</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.9rem;">Cet outil permet de visualiser, analyser et interpréter les données de forages miniers.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Initialisation des variables de session
if 'collars_df' not in st.session_state:
    st.session_state.collars_df = None
    
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
    
if 'lithology_df' not in st.session_state:
    st.session_state.lithology_df = None
    
if 'assays_df' not in st.session_state:
    st.session_state.assays_df = None
    
if 'composites_df' not in st.session_state:
    st.session_state.composites_df = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'hole_id': None,
        'x': None,
        'y': None,
        'z': None,
        'azimuth': None,
        'dip': None,
        'depth': None,
        'lith_from': None,
        'lith_to': None,
        'lithology': None,
        'assay_from': None,
        'assay_to': None,
        'assay_value': None
    }

if 'filtered_holes' not in st.session_state:
    st.session_state.filtered_holes = None

# Page de chargement des données
if page == 'Chargement des données':
    st.markdown('<h2 class="sub-header">Chargement des données</h2>', unsafe_allow_html=True)
    
    # Créer des onglets pour les différents types de données
    tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
    
    # Onglet Collars
    with tabs[0]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de collars</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de collars contiennent les informations sur la position de départ des forages (coordonnées X, Y, Z).</div>', unsafe_allow_html=True)
        
        collars_file = st.file_uploader("Télécharger le fichier CSV des collars", type=['csv'])
        if collars_file is not None:
            try:
                st.session_state.collars_df = pd.read_csv(collars_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.collars_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.collars_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou", 
                                                                             [''] + cols, 
                                                                             index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['x'] = st.selectbox("Colonne X", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['y'] = st.selectbox("Colonne Y", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['z'] = st.selectbox("Colonne Z", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.collars_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Survey
    with tabs[1]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de survey</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de survey contiennent les mesures d\'orientation (azimut, pendage) prises à différentes profondeurs le long du forage.</div>', unsafe_allow_html=True)
        
        survey_file = st.file_uploader("Télécharger le fichier CSV des surveys", type=['csv'])
        if survey_file is not None:
            try:
                st.session_state.survey_df = pd.read_csv(survey_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.survey_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.survey_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Survey)", 
                                                                             [''] + cols, 
                                                                             index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['depth'] = st.selectbox("Colonne profondeur", 
                                                                            [''] + cols,
                                                                            index=0 if len(cols) == 0 else 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['azimuth'] = st.selectbox("Colonne azimut", 
                                                                              [''] + cols,
                                                                              index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['dip'] = st.selectbox("Colonne pendage", 
                                                                          [''] + cols,
                                                                          index=0 if len(cols) == 0 else 1)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Survey)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.survey_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Lithologie
    with tabs[2]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de lithologie</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de lithologie contiennent des informations sur les types de roches rencontrés à différentes profondeurs lors du forage.</div>', unsafe_allow_html=True)
        
        lithology_file = st.file_uploader("Télécharger le fichier CSV des lithologies", type=['csv'])
        if lithology_file is not None:
            try:
                st.session_state.lithology_df = pd.read_csv(lithology_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.lithology_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.lithology_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    hole_id_index = 1 if cols and len(cols) > 0 else 0
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Lithologie)", 
                                                                             [''] + cols, 
                                                                             index=hole_id_index)
                with col2:
                    lith_index = 2 if cols and len(cols) > 1 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['lithology'] = st.selectbox("Colonne de lithologie", 
                                                                               [''] + cols,
                                                                               index=lith_index)
                
                col1, col2 = st.columns(2)
                with col1:
                    lith_from_index = 3 if cols and len(cols) > 2 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['lith_from'] = st.selectbox("Colonne de profondeur début", 
                                                                               [''] + cols,
                                                                               index=lith_from_index)
                with col2:
                    lith_to_index = 4 if cols and len(cols) > 3 else (2 if cols and len(cols) > 1 else 0)
                    st.session_state.column_mapping['lith_to'] = st.selectbox("Colonne de profondeur fin", 
                                                                             [''] + cols,
                                                                             index=lith_to_index)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Lithologie)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.lithology_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Analyses
    with tabs[3]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données d\'analyses</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données d\'analyses contiennent les résultats d\'analyses géochimiques réalisées sur les échantillons de carottes à différentes profondeurs.</div>', unsafe_allow_html=True)
        
        assays_file = st.file_uploader("Télécharger le fichier CSV des analyses", type=['csv'])
        if assays_file is not None:
            try:
                st.session_state.assays_df = pd.read_csv(assays_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.assays_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.assays_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    hole_id_index = 1 if cols and len(cols) > 0 else 0
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Analyses)", 
                                                                             [''] + cols, 
                                                                             index=hole_id_index)
                with col2:
                    value_index = 4 if cols and len(cols) > 3 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['assay_value'] = st.selectbox("Colonne de valeur (par ex. Au g/t)", 
                                                                                 [''] + cols,
                                                                                 index=value_index)
                
                col1, col2 = st.columns(2)
                with col1:
                    from_index = 2 if cols and len(cols) > 1 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['assay_from'] = st.selectbox("Colonne de profondeur début (Analyses)", 
                                                                                [''] + cols,
                                                                                index=from_index)
                with col2:
                    to_index = 3 if cols and len(cols) > 2 else (2 if cols and len(cols) > 1 else 0)
                    st.session_state.column_mapping['assay_to'] = st.selectbox("Colonne de profondeur fin (Analyses)", 
                                                                              [''] + cols,
                                                                              index=to_index)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Analyses)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.assays_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)

# Page d'aperçu des données
elif page == 'Aperçu des données':
    st.markdown('<h2 class="sub-header">Aperçu des données</h2>', unsafe_allow_html=True)
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None and st.session_state.lithology_df is None and st.session_state.assays_df is None:
        st.markdown('<div class="warning-message">⚠️ Aucune donnée n\'a été chargée. Veuillez d\'abord charger des données.</div>', unsafe_allow_html=True)
    else:
        # Section de filtrage
        st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Filtrage des données</h3>', unsafe_allow_html=True)
        
        # Sélectionner le DataFrame à filtrer
        filter_df_choice = st.selectbox("Sélectionner un jeu de données à filtrer", 
                                        ["Collars", "Survey", "Lithologie", "Analyses"],
                                        index=0)
        
        # Récupérer le DataFrame correspondant
        if filter_df_choice == "Collars":
            df_to_filter = st.session_state.collars_df
        elif filter_df_choice == "Survey":
            df_to_filter = st.session_state.survey_df
        elif filter_df_choice == "Lithologie":
            df_to_filter = st.session_state.lithology_df
        else:  # Analyses
            df_to_filter = st.session_state.assays_df
        
        # S'assurer que le DataFrame existe
        if df_to_filter is not None:
            # Interface de filtrage
            st.markdown('<div class="info-card">Filtrez les données en sélectionnant une colonne et en définissant des critères.</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_column = st.selectbox("Colonne à filtrer", 
                                            [''] + df_to_filter.columns.tolist(),
                                            index=0)
            
            with col2:
                filter_type = st.selectbox("Type de filtre", 
                                          ["égal à", "supérieur à", "inférieur à", "contient"],
                                          index=0)
                
                # Mapper les choix aux valeurs internes
                filter_type_map = {
                    "égal à": "equal",
                    "supérieur à": "greater",
                    "inférieur à": "less",
                    "contient": "contains"
                }
                
                # Convertir le choix de l'utilisateur en valeur interne
                filter_type_value = filter_type_map[filter_type]
            
            with col3:
                # Si une colonne a été sélectionnée
                if filter_column:
                    # Afficher les valeurs uniques pour les colonnes catégorielles
                    if pd.api.types.is_object_dtype(df_to_filter[filter_column]) and df_to_filter[filter_column].nunique() < 50:
                        unique_values = df_to_filter[filter_column].unique()
                        filter_value = st.selectbox("Valeur", unique_values)
                    else:
                        filter_value = st.text_input("Valeur")
                else:
                    filter_value = st.text_input("Valeur", disabled=True)
            
            # Bouton pour appliquer le filtre
            if st.button("Appliquer le filtre"):
                if filter_column and filter_value:
                    # Appliquer le filtre
                    filtered_df = filter_dataframe(df_to_filter, filter_column, filter_type_value, filter_value)
                    
                    if not filtered_df.empty:
                        st.markdown(f'<div class="success-message">✅ Filtre appliqué avec succès. {len(filtered_df)} enregistrements filtrés.</div>', unsafe_allow_html=True)
                        
                        # Si on filtre les collars, stocker les forages filtrés
                        if filter_df_choice == "Collars" and st.session_state.column_mapping['hole_id'] in filtered_df.columns:
                            st.session_state.filtered_holes = filtered_df[st.session_state.column_mapping['hole_id']].unique().tolist()
                            st.markdown(f'<div class="info-card">Les forages suivants ont été sélectionnés et seront disponibles pour les visualisations: {", ".join(str(h) for h in st.session_state.filtered_holes)}</div>', unsafe_allow_html=True)
                        
                        # Afficher le résultat
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Bouton pour réinitialiser le filtre
                        if st.button("Réinitialiser le filtre"):
                            st.session_state.filtered_holes = None
                            st.rerun()
                    else:
                        st.markdown('<div class="warning-message">⚠️ Aucun enregistrement ne correspond aux critères de filtrage.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ Veuillez sélectionner une colonne et saisir une valeur pour le filtrage.</div>', unsafe_allow_html=True)
        
        # Créer des onglets pour les différents types de données
        data_tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
        
        # Onglet Collars
        with data_tabs[0]:
            if st.session_state.collars_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de collars</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.collars_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.collars_df.columns:
                        unique_holes = st.session_state.collars_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.collars_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.collars_df, "collars_data.csv", "les données de collars"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de collars n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Survey
        with data_tabs[1]:
            if st.session_state.survey_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de survey</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.survey_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.survey_df.columns:
                        unique_holes = st.session_state.survey_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.survey_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data.csv", "les données de survey"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de survey n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Lithologie
        with data_tabs[2]:
            if st.session_state.lithology_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de lithologie</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.lithology_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.lithology_df.columns:
                        unique_holes = st.session_state.lithology_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                with col3:
                    if 'lithology' in st.session_state.column_mapping and st.session_state.column_mapping['lithology'] in st.session_state.lithology_df.columns:
                        unique_liths = st.session_state.lithology_df[st.session_state.column_mapping['lithology']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_liths}</span><br>Types de lithologies</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.lithology_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.lithology_df, "lithology_data.csv", "les données de lithologie"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de lithologie n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Analyses
        with data_tabs[3]:
            if st.session_state.assays_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données d\'analyses</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.assays_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.assays_df.columns:
                        unique_holes = st.session_state.assays_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                with col3:
                    if 'assay_value' in st.session_state.column_mapping and st.session_state.column_mapping['assay_value'] in st.session_state.assays_df.columns:
                        # Convertir en nombre avant de calculer la moyenne
                        values = pd.to_numeric(st.session_state.assays_df[st.session_state.column_mapping['assay_value']], errors='coerce')
                        avg_value = values.mean()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{avg_value:.2f}</span><br>Valeur moyenne</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.assays_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.assays_df, "assays_data.csv", "les données d'analyses"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée d\'analyses n\'a été chargée.</div>', unsafe_allow_html=True)

# Page de calcul des composites
elif page == 'Composites':
    st.markdown('<h2 class="sub-header">Calcul des composites</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Les composites permettent de regrouper les données d\'analyses en intervalles réguliers pour faciliter l\'interprétation et la modélisation.</div>', unsafe_allow_html=True)
    
    if st.session_state.assays_df is None:
        st.markdown('<div class="warning-message">⚠️ Aucune donnée d\'analyses n\'a été chargée. Veuillez d\'abord charger des données d\'analyses.</div>', unsafe_allow_html=True)
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        assay_from_col = st.session_state.column_mapping['assay_from']
        assay_to_col = st.session_state.column_mapping['assay_to']
        assay_value_col = st.session_state.column_mapping['assay_value']
        
        # Vérifier que les colonnes nécessaires existent dans le DataFrame des analyses
        if not all(col and col in st.session_state.assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
            st.markdown('<div class="warning-message">⚠️ Certaines colonnes nécessaires n\'existent pas dans les données d\'analyses. Veuillez vérifier la sélection des colonnes.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color: #3498DB;">Options de composites</h3>', unsafe_allow_html=True)
            
            # Sélectionner la longueur des composites
            composite_length = st.slider("Longueur des composites (m)", 
                                        min_value=0.5, 
                                        max_value=5.0, 
                                        value=1.0, 
                                        step=0.5)
            
            # Option pour ajouter les coordonnées
            add_coordinates = st.checkbox("Ajouter les coordonnées aux composites", value=True)
            
            # Informations sur l'ajout de coordonnées
            if add_coordinates:
                st.markdown('<div class="info-card" style="margin-top: 10px;">Les coordonnées X, Y, Z seront calculées et ajoutées à chaque composite en utilisant les données de collar et survey.</div>', unsafe_allow_html=True)
            
            # Filtrer par forages
            use_filtered_holes = False
            if st.session_state.filtered_holes:
                use_filtered_holes = st.checkbox("Utiliser uniquement les forages filtrés", value=False)
                if use_filtered_holes:
                    st.markdown(f'<div class="info-card">Seuls les forages suivants seront utilisés pour les composites: {", ".join(str(h) for h in st.session_state.filtered_holes)}</div>', unsafe_allow_html=True)
            
            # Calculer les composites
            if st.button("Calculer les composites"):
                with st.spinner("Calcul des composites en cours..."):
                    try:
                        # Si on utilise les forages filtrés, filter les données d'analyses
                        filtered_assays = st.session_state.assays_df
                        if use_filtered_holes and st.session_state.filtered_holes:
                            filtered_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(st.session_state.filtered_holes)]
                        
                        # Si on demande d'ajouter les coordonnées, vérifier que les données nécessaires sont disponibles
                        include_coordinates = (add_coordinates and 
                                              st.session_state.collars_df is not None and 
                                              st.session_state.survey_df is not None and
                                              all(col and col in st.session_state.collars_df.columns for col in 
                                                  [hole_id_col, st.session_state.column_mapping['x'], 
                                                   st.session_state.column_mapping['y'], st.session_state.column_mapping['z']]) and
                                              all(col and col in st.session_state.survey_df.columns for col in 
                                                  [hole_id_col, st.session_state.column_mapping['depth'], 
                                                   st.session_state.column_mapping['azimuth'], st.session_state.column_mapping['dip']]))
                        
                        if add_coordinates and not include_coordinates:
                            st.markdown('<div class="warning-message">⚠️ Impossible d\'ajouter les coordonnées: données de collar ou survey insuffisantes.</div>', unsafe_allow_html=True)
                        
                        # Filtrer les collars et surveys également si nécessaire
                        filtered_collars = st.session_state.collars_df
                        filtered_surveys = st.session_state.survey_df
                        
                        if use_filtered_holes and st.session_state.filtered_holes:
                            filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(st.session_state.filtered_holes)]
                            filtered_surveys = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(st.session_state.filtered_holes)]
                        
                        st.session_state.composites_df = create_composites(
                            filtered_assays,
                            hole_id_col,
                            assay_from_col,
                            assay_to_col,
                            assay_value_col,
                            composite_length,
                            filtered_collars if include_coordinates else None,
                            filtered_surveys if include_coordinates else None,
                            st.session_state.column_mapping['x'] if include_coordinates else None,
                            st.session_state.column_mapping['y'] if include_coordinates else None,
                            st.session_state.column_mapping['z'] if include_coordinates else None,
                            st.session_state.column_mapping['azimuth'] if include_coordinates else None,
                            st.session_state.column_mapping['dip'] if include_coordinates else None,
                            st.session_state.column_mapping['depth'] if include_coordinates else None
                        )
                        
                        if st.session_state.composites_df is not None and not st.session_state.composites_df.empty:
                            st.markdown(f'<div class="success-message">✅ Composites calculés avec succès. {len(st.session_state.composites_df)} enregistrements générés.</div>', unsafe_allow_html=True)
                            
                            # Afficher les composites
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Résultats des composites</h3>', unsafe_allow_html=True)
                            st.dataframe(st.session_state.composites_df, use_container_width=True)
                            
                            # Lien de téléchargement
                            st.markdown(get_csv_download_link(st.session_state.composites_df, "composites.csv", "les composites"), unsafe_allow_html=True)
                            
                            # Résumé statistique
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Résumé statistique</h3>', unsafe_allow_html=True)
                            
                            # Convertir en flottant pour les calculs statistiques
                            st.session_state.composites_df[assay_value_col] = st.session_state.composites_df[assay_value_col].apply(safe_float)
                            st.write(st.session_state.composites_df[assay_value_col].describe())
                            
                            # Histogramme des valeurs de composites
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Distribution des valeurs</h3>', unsafe_allow_html=True)
                            
                            fig = px.histogram(
                                st.session_state.composites_df, 
                                x=assay_value_col,
                                title=f"Distribution des valeurs de composites ({assay_value_col})",
                                labels={assay_value_col: f'Teneur'},
                                color_discrete_sequence=['#3498DB'],
                                template='plotly_white'
                            )
                            
                            fig.update_layout(
                                xaxis_title=f"Teneur ({assay_value_col})",
                                yaxis_title="Fréquence",
                                title={
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 16, 'color': '#2E4053'}
                                },
                                bargap=0.1
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Comparaison avec les données originales
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Comparaison avec les données originales</h3>', unsafe_allow_html=True)
                            
                            # Convertir les valeurs originales en flottants également
                            filtered_assays[assay_value_col] = filtered_assays[assay_value_col].apply(safe_float)
                            
                            comparison_data = pd.DataFrame({
                                'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                                'Données originales': [
                                    filtered_assays[assay_value_col].mean(),
                                    filtered_assays[assay_value_col].median(),
                                    filtered_assays[assay_value_col].std(),
                                    filtered_assays[assay_value_col].min(),
                                    filtered_assays[assay_value_col].max()
                                ],
                                'Composites': [
                                    st.session_state.composites_df[assay_value_col].mean(),
                                    st.session_state.composites_df[assay_value_col].median(),
                                    st.session_state.composites_df[assay_value_col].std(),
                                    st.session_state.composites_df[assay_value_col].min(),
                                    st.session_state.composites_df[assay_value_col].max()
                                ]
                            })
                            
                            # Créer une figure pour le comparatif
                            fig_comp = go.Figure()
                            
                            # Ajouter une trace pour les données originales
                            fig_comp.add_trace(go.Bar(
                                name='Données originales',
                                x=comparison_data['Statistique'],
                                y=comparison_data['Données originales'],
                                marker_color='#3498DB'
                            ))
                            
                            # Ajouter une trace pour les composites
                            fig_comp.add_trace(go.Bar(
                                name='Composites',
                                x=comparison_data['Statistique'],
                                y=comparison_data['Composites'],
                                marker_color='#2ECC71'
                            ))
                            
                            # Mettre à jour la mise en page
                            fig_comp.update_layout(
                                title={
                                    'text': 'Comparaison des statistiques: Données originales vs Composites',
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 16, 'color': '#2E4053'}
                                },
                                barmode='group',
                                xaxis={'title': 'Statistique'},
                                yaxis={'title': 'Valeur'},
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            # Afficher aussi le tableau des comparaisons
                            st.table(comparison_data.set_index('Statistique').round(3))
                            
                            # Si les coordonnées ont été ajoutées, afficher une visualisation 3D
                            if 'X' in st.session_state.composites_df.columns and 'Y' in st.session_state.composites_df.columns and 'Z' in st.session_state.composites_df.columns:
                                st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Visualisation 3D des composites</h3>', unsafe_allow_html=True)
                                
                                fig_3d = px.scatter_3d(
                                    st.session_state.composites_df,
                                    x='X',
                                    y='Y',
                                    z='Z',
                                    color=assay_value_col,
                                    color_continuous_scale='Viridis',
                                    size=assay_value_col,
                                    size_max=10,
                                    opacity=0.7,
                                    hover_data={
                                        hole_id_col: True,
                                        'From': True,
                                        'To': True,
                                        assay_value_col: ':.2f'
                                    },
                                    title='Distribution spatiale des composites'
                                )
                                
                                fig_3d.update_layout(
                                    scene={
                                        'aspectmode': 'data',
                                        'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.2}},
                                        'xaxis_title': 'X',
                                        'yaxis_title': 'Y',
                                        'zaxis_title': 'Z (Élévation)'
                                    },
                                    margin=dict(l=0, r=0, b=0, t=40),
                                    coloraxis_colorbar=dict(
                                        title=assay_value_col
                                    ),
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.markdown('<div class="error-message">❌ Impossible de calculer les composites avec les données fournies.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Erreur lors du calcul des composites: {str(e)}</div>', unsafe_allow_html=True)

# Page de Carte des Collars
elif page == 'Carte des Collars':
    st.markdown('<h2 class="sub-header">Carte des Collars</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">La carte des collars permet de visualiser la répartition spatiale des forages et de les analyser selon différentes propriétés.</div>', unsafe_allow_html=True)
    
    if st.session_state.collars_df is None:
        st.markdown('<div class="warning-message">⚠️ Les données de collars sont nécessaires pour la carte. Veuillez les charger d\'abord.</div>', unsafe_allow_html=True)
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        x_col = st.session_state.column_mapping['x']
        y_col = st.session_state.column_mapping['z']
        
        # Vérifier que les colonnes nécessaires existent
        if not all(col and col in st.session_state.collars_df.columns for col in [hole_id_col, x_col, y_col]):
            st.markdown('<div class="warning-message">⚠️ Veuillez d\'abord spécifier correctement les colonnes d\'ID de trou et de coordonnées dans la page de chargement des données.</div>', unsafe_allow_html=True)
        else:
            # Options pour la carte
            st.markdown('<h3 style="color: #3498DB;">Options de la carte</h3>', unsafe_allow_html=True)
            
            # Sélectionner quels forages afficher
            if st.session_state.filtered_holes:
                use_filtered_holes = st.checkbox("Utiliser uniquement les forages filtrés", value=False)
                if use_filtered_holes:
                    st.markdown(f'<div class="info-card">Seuls les forages suivants seront affichés: {", ".join(str(h) for h in st.session_state.filtered_holes)}</div>', unsafe_allow_html=True)
            else:
                use_filtered_holes = False
                
            # Récupérer toutes les colonnes du dataframe des collars
            cols = st.session_state.collars_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélectionner une colonne pour la coloration
                color_by = st.selectbox("Colorier par", ['Aucun'] + cols, index=0)
                if color_by == 'Aucun':
                    color_by = None
            
            with col2:
                # Sélectionner une colonne pour la taille
                size_by = st.selectbox("Taille par", ['Aucun'] + cols, index=0)
                if size_by == 'Aucun':
                    size_by = None
            
            # Créer la carte des collars
            try:
                with st.spinner("Création de la carte en cours..."):
                    # Filtrer les données si nécessaire
                    filtered_collars = st.session_state.collars_df
                    if use_filtered_holes and st.session_state.filtered_holes:
                        filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(st.session_state.filtered_holes)]
                    
                    # Créer la carte
                    collar_map = create_collar_map(
                        filtered_collars, 
                        hole_id_col, 
                        x_col, 
                        y_col, 
                        color_by, 
                        size_by
                    )
                    
                    if collar_map:
                        st.plotly_chart(collar_map, use_container_width=True)
                    else:
                        st.markdown('<div class="error-message">❌ Impossible de créer la carte avec les données fournies.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors de la création de la carte: {str(e)}</div>', unsafe_allow_html=True)

# Page de Strip Logs
elif page == 'Strip Logs':
    st.markdown('<h2 class="sub-header">Strip Logs des forages</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Les strip logs permettent de visualiser graphiquement les données de lithologie et d\'analyses le long d\'un forage.</div>', unsafe_allow_html=True)
    
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.markdown('<div class="warning-message">⚠️ Les données de collars et de survey sont nécessaires pour les strip logs. Veuillez les charger d\'abord.</div>', unsafe_allow_html=True)
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        depth_col = st.session_state.column_mapping['depth']
        
        # Vérifier que les colonnes nécessaires existent
        if not hole_id_col or not depth_col or hole_id_col not in st.session_state.collars_df.columns or hole_id_col not in st.session_state.survey_df.columns or depth_col not in st.session_state.survey_df.columns:
            st.markdown('<div class="warning-message">⚠️ Veuillez d\'abord spécifier correctement les colonnes d\'ID de trou et de profondeur dans la page de chargement des données.</div>', unsafe_allow_html=True)
        else:
            # Sélection du forage à afficher
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            # Filtrer la liste des forages si nécessaire
            if st.session_state.filtered_holes:
                use_filtered_holes = st.checkbox("Utiliser uniquement les forages filtrés", value=False)
                if use_filtered_holes:
                    all_holes = [h for h in all_holes if h in st.session_state.filtered_holes]
                    st.markdown(f'<div class="info-card">Sélection limitée aux forages filtrés ({len(all_holes)} forages).</div>', unsafe_allow_html=True)
            
            if not all_holes:
                st.markdown('<div class="warning-message">⚠️ Aucun forage trouvé dans les données.</div>', unsafe_allow_html=True)
            else:
                selected_hole = st.selectbox("Sélectionner un forage", all_holes)
                
                if selected_hole:
                    # Récupérer les informations du forage sélectionné
                    selected_collar = st.session_state.collars_df[st.session_state.collars_df[hole_id_col] == selected_hole]
                    selected_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col] == selected_hole]
                    
                    if not selected_survey.empty:
                        # Afficher les informations du forage
                        collar_info = selected_collar.iloc[0]
                        
                        st.markdown(f'<h3 style="color: #3498DB;">Informations sur le forage {selected_hole}</h3>', unsafe_allow_html=True)
                        
                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            x_col = st.session_state.column_mapping['x']
                            y_col = st.session_state.column_mapping['y']
                            z_col = st.session_state.column_mapping['z']
                            
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Coordonnées</p>', unsafe_allow_html=True)
                            
                            if x_col and x_col in selected_collar.columns:
                                st.write(f"X: {safe_float(collar_info[x_col]):.2f}")
                            if y_col and y_col in selected_collar.columns:
                                st.write(f"Y: {safe_float(collar_info[y_col]):.2f}")
                            if z_col and z_col in selected_collar.columns:
                                st.write(f"Z: {safe_float(collar_info[z_col]):.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with info_cols[1]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            max_depth = safe_float(selected_survey[depth_col].max())
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Profondeur</p>', unsafe_allow_html=True)
                            st.write(f"Profondeur maximale: {max_depth:.2f} m")
                            
                            # Infos supplémentaires si lithologie disponible
                            if st.session_state.lithology_df is not None:
                                lith_col = st.session_state.column_mapping['lithology']
                                if hole_id_col in st.session_state.lithology_df.columns and lith_col and lith_col in st.session_state.lithology_df.columns:
                                    selected_litho = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col] == selected_hole]
                                    if not selected_litho.empty:
                                        unique_liths = selected_litho[lith_col].nunique()
                                        st.write(f"Nombre de lithologies: {unique_liths}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with info_cols[2]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            # Infos supplémentaires si analyses disponibles
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Analyses</p>', unsafe_allow_html=True)
                            if st.session_state.assays_df is not None:
                                assay_value_col = st.session_state.column_mapping['assay_value']
                                if hole_id_col in st.session_state.assays_df.columns and assay_value_col and assay_value_col in st.session_state.assays_df.columns:
                                    selected_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col] == selected_hole]
                                    if not selected_assays.empty:
                                        # Convertir en nombres
                                        selected_assays[assay_value_col] = selected_assays[assay_value_col].apply(safe_float)
                                        avg_value = selected_assays[assay_value_col].mean()
                                        max_value = selected_assays[assay_value_col].max()
                                        st.write(f"Valeur moyenne: {avg_value:.2f}")
                                        st.write(f"Valeur maximale: {max_value:.2f}")
                                else:
                                    st.write("Aucune donnée d'analyse disponible")
                            else:
                                st.write("Aucune donnée d'analyse disponible")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Créer et afficher le strip log
                        try:
                            with st.spinner("Création du strip log en cours..."):
                                strip_log_image = create_strip_log(
                                    selected_hole,
                                    st.session_state.collars_df,
                                    st.session_state.survey_df,
                                    st.session_state.lithology_df,
                                    st.session_state.assays_df,
                                    hole_id_col,
                                    depth_col,
                                    st.session_state.column_mapping['lith_from'],
                                    st.session_state.column_mapping['lith_to'],
                                    st.session_state.column_mapping['lithology'],
                                    st.session_state.column_mapping['assay_from'],
                                    st.session_state.column_mapping['assay_to'],
                                    st.session_state.column_mapping['assay_value']
                                )
                            
                            if strip_log_image:
                                st.image(strip_log_image, caption=f"Strip Log du forage {selected_hole}", use_column_width=True)
                                
                                # Téléchargement de l'image
                                download_col1, download_col2 = st.columns([1, 3])
                                with download_col1:
                                    btn = st.download_button(
                                        label="Télécharger le strip log",
                                        data=strip_log_image,
                                        file_name=f"strip_log_{selected_hole}.png",
                                        mime="image/png"
                                    )
                            else:
                                st.markdown('<div class="error-message">❌ Impossible de créer le strip log avec les données fournies.</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="error-message">❌ Erreur lors de la création du strip log: {str(e)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-message">❌ Aucune donnée de survey trouvée pour le forage {selected_hole}.</div>', unsafe_allow_html=True)

# Page de visualisation 3D
elif page == 'Visualisation 3D':
    st.markdown('<h2 class="sub-header">Visualisation 3D des forages</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">La visualisation 3D permet de visualiser les trajectoires des forages dans l\'espace et les données associées.</div>', unsafe_allow_html=True)
    
    # Vérifier si les données nécessaires ont été chargées
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.markdown('<div class="warning-message">⚠️ Les données de collars et de survey sont nécessaires pour la visualisation 3D. Veuillez les charger d\'abord.</div>', unsafe_allow_html=True)
    else:
        # Vérifier si les colonnes nécessaires ont été spécifiées
        hole_id_col = st.session_state.column_mapping['hole_id']
        x_col = st.session_state.column_mapping['x']
        y_col = st.session_state.column_mapping['y']
        z_col = st.session_state.column_mapping['z']
        azimuth_col = st.session_state.column_mapping['azimuth']
        dip_col = st.session_state.column_mapping['dip']
        depth_col = st.session_state.column_mapping['depth']
        
        required_cols_exist = (
            hole_id_col and hole_id_col in st.session_state.collars_df.columns and hole_id_col in st.session_state.survey_df.columns and
            x_col and x_col in st.session_state.collars_df.columns and
            y_col and y_col in st.session_state.collars_df.columns and
            z_col and z_col in st.session_state.collars_df.columns and
            azimuth_col and azimuth_col in st.session_state.survey_df.columns and
            dip_col and dip_col in st.session_state.survey_df.columns and
            depth_col and depth_col in st.session_state.survey_df.columns
        )
        
        if not required_cols_exist:
            st.markdown('<div class="warning-message">⚠️ Certaines colonnes requises n\'ont pas été spécifiées ou n\'existent pas dans les données. Veuillez vérifier la sélection des colonnes dans l\'onglet \'Chargement des données\'.</div>', unsafe_allow_html=True)
        else:
            # Options pour la visualisation
            st.markdown('<h3 style="color: #3498DB;">Options de visualisation</h3>', unsafe_allow_html=True)
            
            # Sélection des forages à afficher
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            # Ajouter l'option d'utiliser les forages filtrés
            if st.session_state.filtered_holes:
                use_filtered_holes = st.checkbox("Utiliser uniquement les forages filtrés", value=False)
                if use_filtered_holes:
                    available_holes = [h for h in all_holes if h in st.session_state.filtered_holes]
                    st.markdown(f'<div class="info-card">Sélection limitée aux forages filtrés ({len(available_holes)} forages).</div>', unsafe_allow_html=True)
                else:
                    available_holes = all_holes
            else:
                available_holes = all_holes
                
            if not available_holes:
                st.markdown('<div class="warning-message">⚠️ Aucun forage trouvé dans les données.</div>', unsafe_allow_html=True)
            else:
                selected_holes = st.multiselect("Sélectionner les forages à afficher", available_holes, default=available_holes[:min(5, len(available_holes))])
                
                # Options additionnelles
                option_cols = st.columns(2)
                with option_cols[0]:
                    show_lithology = st.checkbox("Afficher la lithologie", value=True if st.session_state.lithology_df is not None else False)
                with option_cols[1]:
                    show_assays = st.checkbox("Afficher les teneurs", value=True if st.session_state.assays_df is not None else False)
                
                # Sélection du type de visualisation 3D
                viz_method = st.radio("Méthode de visualisation", ["Plotly (Standard)", "PyVista (Avancé)"], index=0)
                
                # Filtrer les données selon les forages sélectionnés
                if selected_holes:
                    filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(selected_holes)]
                    filtered_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(selected_holes)]
                    
                    # Filtrer lithology et assays si nécessaire
                    filtered_lithology = None
                    if show_lithology and st.session_state.lithology_df is not None:
                        # Vérifier que hole_id_col existe dans lithology_df avant de filtrer
                        if hole_id_col in st.session_state.lithology_df.columns:
                            filtered_lithology = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col].isin(selected_holes)]
                        else:
                            st.markdown(f'<div class="warning-message">⚠️ La colonne {hole_id_col} n\'existe pas dans les données de lithologie.</div>', unsafe_allow_html=True)
                    
                    filtered_assays = None
                    if show_assays and st.session_state.assays_df is not None:
                        # Vérifier que hole_id_col existe dans assays_df avant de filtrer
                        if hole_id_col in st.session_state.assays_df.columns:
                            filtered_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(selected_holes)]
                        else:
                            st.markdown(f'<div class="warning-message">⚠️ La colonne {hole_id_col} n\'existe pas dans les données d\'analyses.</div>', unsafe_allow_html=True)
                    
                    try:
                        with st.spinner("Création de la visualisation 3D en cours..."):
                            if viz_method == "Plotly (Standard)":
                                # Utiliser la méthode Plotly pour la visualisation 3D
                                fig_3d = create_drillhole_3d_plot(
                                    filtered_collars, 
                                    filtered_survey, 
                                    filtered_lithology, 
                                    filtered_assays,
                                    hole_id_col=hole_id_col,
                                    x_col=x_col,
                                    y_col=y_col,
                                    z_col=z_col,
                                    azimuth_col=azimuth_col,
                                    dip_col=dip_col,
                                    depth_col=depth_col,
                                    lith_from_col=st.session_state.column_mapping['lith_from'],
                                    lith_to_col=st.session_state.column_mapping['lith_to'],
                                    lith_col=st.session_state.column_mapping['lithology'],
                                    assay_from_col=st.session_state.column_mapping['assay_from'],
                                    assay_to_col=st.session_state.column_mapping['assay_to'],
                                    assay_value_col=st.session_state.column_mapping['assay_value']
                                )
                                
                                if fig_3d:
                                    st.plotly_chart(fig_3d, use_container_width=True, height=800)
                                else:
                                    st.markdown('<div class="error-message">❌ Impossible de créer la visualisation 3D avec les données fournies.</div>', unsafe_allow_html=True)
                            else:
                                # Utiliser PyVista pour la visualisation 3D
                                if not PYVISTA_AVAILABLE:
                                    st.markdown('<div class="error-message">❌ PyVista n\'est pas disponible. Veuillez installer les dépendances nécessaires pour utiliser cette fonctionnalité.</div>', unsafe_allow_html=True)
                                else:
                                    pv_image_path, error_msg = create_pyvista_3d(
                                        filtered_collars, 
                                        filtered_survey, 
                                        filtered_lithology, 
                                        filtered_assays,
                                        hole_id_col=hole_id_col,
                                        x_col=x_col,
                                        y_col=y_col,
                                        z_col=z_col,
                                        azimuth_col=azimuth_col,
                                        dip_col=dip_col,
                                        depth_col=depth_col,
                                        lith_from_col=st.session_state.column_mapping['lith_from'],
                                        lith_to_col=st.session_state.column_mapping['lith_to'],
                                        lith_col=st.session_state.column_mapping['lithology'],
                                        assay_from_col=st.session_state.column_mapping['assay_from'],
                                        assay_to_col=st.session_state.column_mapping['assay_to'],
                                        assay_value_col=st.session_state.column_mapping['assay_value'],
                                        selected_holes=selected_holes,
                                        show_lithology=show_lithology,
                                        show_assays=show_assays
                                    )
                                    
                                    if pv_image_path and os.path.exists(pv_image_path):
                                        st.image(pv_image_path, caption="Visualisation 3D des forages avec PyVista", use_column_width=True)
                                        
                                        # Lien de téléchargement
                                        with open(pv_image_path, "rb") as f:
                                            image_bytes = f.read()
                                            
                                        st.download_button(
                                            label="Télécharger l'image 3D",
                                            data=image_bytes,
                                            file_name="forage_3d.png",
                                            mime="image/png"
                                        )
                                        
                                        # Nettoyer le fichier temporaire
                                        try:
                                            os.remove(pv_image_path)
                                        except:
                                            pass
                                    elif error_msg:
                                        st.markdown(f'<div class="error-message">❌ {error_msg}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="error-message">❌ Erreur lors de la création de la visualisation 3D avec PyVista.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Erreur lors de la création de la visualisation 3D: {str(e)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-card">Veuillez sélectionner au moins un forage à afficher.</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #F8F9F9; border-top: 1px solid #E5E7E9;">
    <p style="font-size: 0.8rem; color: #7F8C8D;">© 2025 Didier Ouedraogo, P.Geo. | Mining Geology Data Application</p>
</div>
""", unsafe_allow_html=True)